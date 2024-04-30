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
import torch
import xarray as xr


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
        # If input is a folder, find all .HDF5 files.
        if isinstance(inputs, Path) and path.is_dir():
            self.files = sorted(list(path.glob("**/*.HDF5")))
        else:
            # If input is a list assume its list of input files.
            if isinstance(inputs, list):
                self.files = inputs
            else:
                # If input is a single file, turn it into a list.
                self.files = [inputs]
        self.config = config


    def load_data(self, input_file: Path) -> Tuple[Dict[str, torch.tensor], Path, xr.Dataset]:
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
        l1c_file = L1CFile(input_file)
        sensor = l1c_file.sensor
        input_data = run_preprocessor(input_file, sensor)

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
        tbs = input_data.brightness_temperatures.data
        tbs[tbs < 0] = np.nan
        tbs_full = np.nan * np.zeros((tbs.shape[:2] +(15,)), dtype=np.float32)
        tbs_full[..., sensor.gprof_channels] = tbs

        # Loading of angles and ancillary data included here for completeness
        # but ingnored by the non-ancillary-data models.

        angs = input_data.earth_incidence_angle.data
        angs[angs < -100] = np.nan
        angs_full = np.nan * np.zeros((angs.shape[:2] +(15,)), dtype=np.float32)
        angs_full[..., sensor.gmi_channels] = angs

        anc = np.stack([input_data[var] for var in ANCILLARY_VARIABLES], -1)

        if self.config == "1d":
            return {
                "brightness_temperatures": torch.tensor(tbs_full.reshape(-1, 15)),
                "viewing_angles": torch.tensor(angs_full.reshape(-1, 15)),
                "ancillary_data": torch.tensor(anc.reshape(-1, 8)),
            }, filename, input_data

        tbs_full = np.transpose(tbs_full, (2, 0, 1))
        angs_full = np.transpose(angs_full, (2, 0, 1))
        anc = np.transpose(anc, (2, 0, 1))

        # Filename and input_data are returned to reuse them in 'finalize_results'.
        return {
            "brightness_temperatures": torch.tensor(tbs_full),
            "viewing_angles": torch.tensor(angs_full),
            "ancillary_data": torch.tensor(anc),
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
            filename: Path,
            preprocessor_data: xr.Dataset
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
        data = preprocessor_data.copy()
        shape = (data.scans.size, data.pixels.size)

        dims = ("levels", "scans", "pixels")

        for var, tensor in results.items():
            if var == "surface_precip_terciles":
                data["surface_precip_1st_tercile"] = (
                    ("scans", "pixels"), tensor[0].numpy()
                )
                data["surface_precip_2nd_tercile"] = (
                    ("scans", "pixels"),
                    tensor[1].numpy()
                )
            else:
                dims_v = dims[-tensor.ndim():]
                data[var] = (dims_v, tensor.numpy())

        output_filename = filename.replace("1C-R", "2A").replace("1C", "2A")
        output_filename = output_filename.with_suffix(".nc")
        return data
