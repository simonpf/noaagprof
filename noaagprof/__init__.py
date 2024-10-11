"""
noaagprof
=========

The 'noaagprof' package provides site-specific functionality to run AMSR2 GPROF-NN retrievals
for NOAA.
"""
from pathlib import Path
from typing import Dict, List, Union, Tuple

from pansat import Granule
from pansat.products.satellite.noaa.gaasp import l1b_gcomw1_amsr2
import numpy as np
import torch
import xarray as xr


class InputLoader:
    """
    The input loader class identifies input files and provides an interator over the input
    data from each detected file.

    Attributes:
        files: A list of the input files to be loaded.
    """
    def __init__(
            self,
            inputs: Union[str, Path, List[str], List[Path]],
    ):
        """
        Args:
            inputs: A single string or Path object, or lists thereof, pointing to either a
                collection of input files or an input folder.
        """
        if isinstance(inputs, str):
            inputs = Path(inputs)

        # If input is a folder, find all .HDF5 files.
        if isinstance(inputs, Path) and inputs.is_dir():
            self.files = sorted(list(inputs.glob("**/*.HDF5")))
        else:
            # If input is a list assume its list of input files.
            if isinstance(inputs, list):
                self.files = inputs
            else:
                # If input is a single file, turn it into a list.
                self.files = [inputs]


    def load_data(self, input_file: Path) -> Tuple[Dict[str, torch.tensor], str, xr.Dataset]:
        """
        Load input data for a given file.

        Loads the input data from a NOAA AMSR2 L1B file into a brightness temperature tensor in the
        expected format.

        Args:
            input_file: A path object pointing to an input file.

        Return:
            A tuple ``(input_data, input_filename, aux_data)``, where:
                - ``input_data`` is a dictionary containing a single key-value pair "brightness_temperatures"
                  containing a torch.Tensor with all observations in the given file as a 3D tensor (if config is '3D')
                  or 1D tensor (if config is '1D').
                - ``input_filename``: Is a Path object pointing to the input file.
                - ``aux_data``: Contains auxiliary data such as geolocation coordinates
                  from the input file.

        Note:
            The inference function of 'pytorch_retrieve' will pass ``input_filename`` and ``preprocessor_data``
            to the 'finalize_results' method defined bleow after performing the inference. They are used their
            to determine the output filename and to include ancillary data in the retrieval results.
        """
        if isinstance(input_file, Granule):
            l1b_data = input_file.open()
            filename = input_file.file_record.filename
        else:
            l1b_data = l1b_gcomw1_amsr2.open(input_file)
            filename = input_file

        lats = l1b_data.latitude_s2.data
        lons = l1b_data.longitude_s2.data

        tbs_s1 = np.repeat(l1b_data.tbs_s1.data, 2, axis=1)
        tbs_s2 = l1b_data.tbs_s2.data
        tbs_s3 = l1b_data.tbs_s3.data
        sc_lon = l1b_data.spacecraft_longitude.data
        sc_lat = l1b_data.spacecraft_latitude.data
        sc_alt = l1b_data.spacecraft_altitude.data
        tbs = np.concatenate((tbs_s1, tbs_s2, tbs_s3), axis=-1)
        scan_time = l1b_data.scan_time.data

        # Upsample fields
        n_scans, n_pixels, n_channels = tbs.shape

        lons_hr = np.zeros_like(lons, shape=(2 * n_scans - 1, n_pixels))
        lons_hr[::2] = lons
        lons_hr[1::2] = 0.5 * (lons[:-1] + lons[1:])
        lats_hr = np.zeros_like(lats, shape=(2 * n_scans - 1, n_pixels))
        lats_hr[::2] = lats
        lats_hr[1::2] = 0.5 * (lats[:-1] + lats[1:])
        tbs_hr = np.zeros_like(tbs, shape=(2 * n_scans - 1, n_pixels, n_channels))
        tbs_hr[::2] = tbs
        tbs_hr[1::2] = 0.5 * (tbs[:-1] + tbs[1:])
        scan_time_hr = np.zeros_like(scan_time, shape=(2 * n_scans - 1))
        scan_time_hr[::2] = scan_time
        scan_time_hr[1::2] = scan_time[1:] + 0.5 * (scan_time[1:] - scan_time[:-1])

        sc_lon_hr = np.zeros_like(sc_lon, shape=(2 * n_scans - 1))
        sc_lon_hr[::2] = sc_lon
        sc_lon_hr[1::2] = 0.5 * (sc_lon[:-1] + sc_lon[1:])
        sc_lat_hr = np.zeros_like(sc_lat, shape=(2 * n_scans - 1))
        sc_lat_hr[::2] = sc_lat
        sc_lat_hr[1::2] = 0.5 * (sc_lat[:-1] + sc_lat[1:])
        sc_alt_hr = np.zeros_like(sc_alt, shape=(2 * n_scans - 1))
        sc_alt_hr[::2] = sc_alt
        sc_alt_hr[1::2] = 0.5 * (sc_alt[:-1] + sc_alt[1:])

        input_data = xr.Dataset({
            "latitude": (("scan", "pixel"), lats_hr),
            "longitude": (("scan", "pixel"), lons_hr),
            "observations": (("scan", "pixel", "channels"), tbs_hr),
            "spacecraft_longitude": (("scan"), sc_lon_hr),
            "spacecraft_latitude": (("scan"), sc_lat_hr),
            "spacecraft_altitude": (("scan"), sc_alt_hr),
            "scan_time": (("scan"), scan_time_hr.astype("datetime64[ns]"))
        })

        tbs = input_data.observations.data
        tbs[tbs < 0] = np.nan
        tbs = np.transpose(tbs, (2, 0, 1))

        return {
            "amsr2": torch.tensor(tbs)[None],
        }, filename, input_data

    def __len__(self):
        """
        The number of input files found.
        """
        return len(self.files)

    def __iter__(self):
        """
        Iterate over all input files.
        """
        for file in self.files:
            yield self.load_data(file)

    def finalize_results(
            self,
            results: Dict[str, torch.Tensor],
            filename: str,
            input_data: xr.Dataset
    ) -> xr.Dataset:
        """
        This function is called after inference has been performed on all input data from a given file.
        The purpose of this function is to define the output name and customize the format and content of
        the retrieval output.

        Args:
            results: A dictionary mapping retrieval output names to corresponding
                tensors containing the results.
            filename: The filename as returned by the 'load_data' method.
            input_data: The input data as returned by the 'load_data' method.

        Return:
            A tuple ``(results, output_filename)`` containing the potentially updated retrieval results and
            the ``output_filename`` to use to store the retrieval results.
        """
        data = input_data.copy()
        shape = (data.scan.size, data.pixel.size)

        dims = ("levels", "scan", "pixel")

        for var, tensor in results.items():

            # Discard dummy dimensions.
            tensor = tensor.squeeze()

            if var == "surface_precip_terciles":
                data["surface_precip_1st_tercile"] = (
                    ("scan", "pixel"), tensor[0].numpy()
                )
                data["surface_precip_1st_tercile"].encoding = {"dtype": "float32", "zlib": True}
                data["surface_precip_2nd_tercile"] = (
                    ("scan", "pixel"),
                    tensor[1].numpy()
                )
                data["surface_precip_2nd_tercile"].encoding = {"dtype": "float32", "zlib": True}
            else:
                dims_v = dims[-tensor.dim():]
                if tensor.shape[0] < 28:
                    dims_v = ("classes",) + dims_v[1:]

                data[var] = (dims_v, tensor.numpy())
                # Use compressiong to keep file size reasonable.
                data[var].encoding = {"dtype": "float32", "zlib": True}


        # Quick and dirty way to transform 1C filename to 2A filename
        output_filename = (
            filename.replace("1B", "2A")
            .replace("1B", "2A")
            .replace("HDF5", "nc")
        )

        # Return results as xr.Dataset and filename to use to save data.
        return data, output_filename
