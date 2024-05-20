"""
noaagprof
=========

The 'noaagprof' package provides site-specific functionality to run AMSR2 GPROF-NN retrievals
for NOAA.
"""
from pathlib import Path
from typing import Dict, List, Union, Tuple

from gprof_nn.data.preprocessor import run_preprocessor
from gprof_nn.data.l1c import L1CFile
from gprof_nn import sensors
import numpy as np
import torch
import xarray as xr
import h5py 

def read_noaa_gcom_l1b_file(file_path):
# A function to read in and resample GCOMW1-AMSR2 TBs

  # read in the data
  with h5py.File(file_path, 'r') as file:

    # Read only the necessary vars
    dataset_vars=                             \
    ["Brightness Temperature (10.7GHz,V)"   , \
     "Brightness Temperature (10.7GHz,H)"   , \
     "Brightness Temperature (18.7GHz,V)"   , \
     "Brightness Temperature (18.7GHz,H)"   , \
     "Brightness Temperature (23.8GHz,V)"   , \
     "Brightness Temperature (23.8GHz,H)"   , \
     "Brightness Temperature (36.5GHz,V)"   , \
     "Brightness Temperature (36.5GHz,H)"   , \
     "Brightness Temperature (89.0GHz-A,V)" , \
     "Brightness Temperature (89.0GHz-A,H)" , \
     "Latitude of Observation Point for 89A", \
     "Longitude of Observation Point for 89A"]

    in_count=0 # channel counter, for indexing purposes

    # loop over variables to read from l1b and store into arrays
    for var in dataset_vars:
      if var in file:
        data_raw = file[var]
      else:
        print("Dataset", var, "not found in the file.")

      # Check if the dataset has a scale factor attribute
      if 'SCALE FACTOR' in data_raw.attrs:
        scale_factor = data_raw.attrs['SCALE FACTOR']

        # Apply the scale factor to obtain the original data
        data = data_raw * scale_factor

      # declare output array. Use the 1st channel to determine the size (i.e., # of scans)
      if var == dataset_vars[0]:
        tbs_out  = np.empty((data.shape[0],486,10),dtype=float)
        lon      = np.empty((data.shape[0],486)   ,dtype=float)
        lat      = np.empty((data.shape[0],486)   ,dtype=float)

      # for all low freq. channels (0-7) replicate pixels along the scanline to match the high freq. sampling
      if in_count <= 7:
        data = np.repeat(data, 2, axis=1)

      # Store data into the output arrays
      if in_count <= 9:
        tbs_out [:,:,in_count]=np.array(data)

      if in_count == 10:
        lat[:,:]=np.array(data)

      if in_count == 11:
        lon[:,:]=np.array(data)

      # increase channel count
      in_count=in_count+1

  return tbs_out, lat, lon


class InputLoader:
    """
    The input loader class identifies input files and provides an interator over the input
    data from each detected file.

    Attributes:
        files: A list of the input files to be loaded.
        config: A string specifying the retrieval config: '1D' or '3D'.
    """
    def __init__(
            self,
            inputs: Union[str, Path, List[str], List[Path]],
            config: str,
    ):
        """
        Args:
            inputs: A single string or Path object, or lists thereof, pointing to either a
                collection of input files or an input folder.
            config: A string defining the retrieval configuration: '1D' or '3D'
        """
        if isinstance(inputs, str):
            inputs = Path(inputs)

        # If input is a folder, find all .HDF5 files.
        if isinstance(inputs, Path) and inputs.is_dir():
           #self.files = sorted(list(inputs.glob("**/*.HDF5")))
            self.files = sorted(list(inputs.glob("**/*.h5")))
        else:
            # If input is a list assume its list of input files.
            if isinstance(inputs, list):
                self.files = inputs
            else:
                # If input is a single file, turn it into a list.
                self.files = [inputs]
        self.config = config


    def load_data(self, input_file: Path) -> Tuple[Dict[str, torch.tensor], str, xr.Dataset]:
        """
        Load input data for a given file.

        Runs the preprocessor on the given L1C file and loads the retrieval input data
        from the results.

        Args:
            input_file: A path object pointing to an input file.

        Return:
            A tuple ``(input_data, input_filename, preprocessor_data)``, where:
                - ``input_data`` is a dictionary containing a single key-value pair "brightness_temperatures"
                  containing a torch.Tensor with all observations in the given file as a 3D tensor (if config is '3D')
                  or 2D tensor (if config is '2D').
                - ``input_filename``: Is a Path object pointing to the input file.
                - ``preprocessor_data``: Is

        Note:
            The inference function of 'pytorch_retrieve' will pass ``input_filename`` and ``preprocessor_data``
            to the 'finalize_results' method defined bleow after performing the inference. They are used their
            to determine the output filename and to include ancillary data in the retrieval results.
        """

        input_file = Path(input_file)
        filename = input_file.name

        # Tbs and angles must be expanded to the 15 GPROF channels.
        # For AMSR2 those are frequencies
        #
        #  Index | Freq [GHz] |  Pol
        #  ------|-----------|-----
        #      0 |  10.65     |  "V"
        #      1 |  10.65     |  "H"
        #      2 |  18.7      |  "V"
        #      3 |  18.7      |  "H"
        #      4 |  23.8      |  "V"
        #      5 |  23.8      |  "H"
        #      6 |  36.5      |  "V"
        #      7 |  36.5      |  "H"
        #      8 |  89        |  "V"
        #      9 |  89        |  "H"
        #

        tbs, lat, lon = read_noaa_gcom_l1b_file(input_file)     
        tbs[tbs < 0]  = np.nan
        tbs_full      = np.nan * np.zeros((tbs.shape[:2] +(15,)), dtype=np.float32)
        tbs_full[..., :10] = tbs  # populate only the first 10 channels (AMSR2)

        # Order of dimensions should be [batch, channels, scans, pixel].
        # [None] adds dummy batch dimension.
        tbs_full = np.transpose(tbs_full, (2, 0, 1))[None]

        return {
            "brightness_temperatures": torch.tensor(tbs_full)
        }, filename, lat, lon

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
            lat, lon 
           # preprocessor_data: xr.Dataset
    ) -> xr.Dataset:


        """
        This function is called after inference has been performed on all input data from a given file.
        The purpose of this function is to define the output name and customize the format and content of
        the retrieval output.

        Args:
            results: A dictionary mapping retrieval output names to corresponding
                tensors containing the results.
            filename: The filename as returned by the 'load_data' method.
            preprocessor_data: The preprocessor data as returned by the 'load_data' method.

        Return:
            A tuple ``(results, output_filename)`` containing the potentially updated retrieval results and
            the ``output_filename`` to use to store the retrieval results.
        """

       # data = preprocessor_data.copy()
        data = xr.Dataset({
            'latitude' :(('scans','pixels'),lat) ,
            'longitude':(('scans','pixels'),lon) })

       # shape = (data.scans.size, data.pixels.size)

        dims = ("levels", "scans", "pixels")

        for var, tensor in results.items():

            # Discard dummy dimensions.
            tensor = tensor.squeeze()

            if var == "surface_precip_terciles":
                data["surface_precip_1st_tercile"] = (
                    ("scans", "pixels"), tensor[0].numpy()
                )
                data["surface_precip_1st_tercile"].encoding = {"dtype": "float32", "zlib": True}
                data["surface_precip_2nd_tercile"] = (
                    ("scans", "pixels"),
                    tensor[1].numpy()
                )
                data["surface_precip_2nd_tercile"].encoding = {"dtype": "float32", "zlib": True}
            else:
                dims_v = dims[-tensor.dim():]
                data[var] = (dims_v, tensor.numpy())
                # Use compressiong to keep file size reasonable.
                data[var].encoding = {"dtype": "float32", "zlib": True}


        # Quick and dirty way to transform 1C filename to 2A filename
        output_filename = (
            filename.replace("1C-R", "2A")
            .replace("1C", "2A")
            .replace("HDF5", "nc")
            .replace("h5", "nc")
            .replace("GAASP-L1B", "noaagprof_AMSR2_PRECIP_v4r0")
        )

        # Return results as xr.Dataset and filename to use to save data.
        return data, output_filename
