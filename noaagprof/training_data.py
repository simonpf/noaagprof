"""
noaagprof.training_data
=======================

This module defines a PyTorch data loader for loading the MRMS and CMB
 collocations used to traing the NOAAGPROF neural network.
"""
from datetime import datetime
import logging
from typing import Any, Dict, List, Tuple, Union
import os
from pathlib import Path

import hdf5plugin
import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr


LOGGER = logging.getLogger(__name__)


def extract_time(path: Path) -> datetime:
    """
    Extract time from training file name.

    Args:
        path: A Path object pointing to a NOAAGPROF training file.

    Return:
        A datetime.datetime object representing the time stamp of
        the given training file.

    """
    date_str = path.name.split("_")[-1][:-3]
    return datetime.strptime(date_str, "%Y%m%d%H%M%S")


TARGETS = {
    "surface_precip": (),
    "precipitation_type": (),
    "total_water_content": (28,),
    "rain_water_content": (28,),
    "rain_water_path": (),
    "snow_water_path": (),
    "snow_water_content": (28,),
    "latent_heating": (28,),
}


class TrainingDataset(Dataset):
    """
    Dataset class to load the NOAAGPROF training data.
    """
    def __init__(
        self,
        paths: Union[Path, List[Path]],
        augment: bool = True,
    ):
        """
        Args:
            paths: A single Path object or a list of Path objects pointing to
                directories containing the folders with the input and target data.
            augment: Whether or not to apply random flips and transpose to the
                input samples.
        """
        super().__init__()
        self.augment = augment

        if not isinstance(paths, list):
            paths = [paths]

        all_input_files = []
        all_target_files = []

        for path in paths:
            path = Path(path)
            input_files = sorted(list((path / "noaa").glob("*.nc")))
            input_times = list(map(extract_time, input_files))
            input_files = dict(list(zip(input_times, input_files)))
            target_files = sorted(list((path / "target").glob("*.nc")))
            target_times = list(map(extract_time, target_files))
            target_files = dict(list(zip(target_times, target_files)))

            matched_times = sorted(list(set(input_files) & set(target_files)))
            all_input_files += [input_files[time] for time in matched_times]
            all_target_files += [target_files[time] for time in matched_times]

        self.input_files = np.array(all_input_files)
        self.target_files = np.array(all_target_files)

        self.worker_init_fn(0)


    def worker_init_fn(self, w_id: int) -> None:
        """
        Seeds the dataset loader's random number generator.
        """
        seed = int.from_bytes(os.urandom(4), "big") + w_id
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        """
        The number of samples in the dataset.
        """
        return len(self.input_files)

    def __getitem__(self, ind: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Load sample from dataset.
        """
        try:
            with xr.open_dataset(self.input_files[ind]) as data:
                inpt_data = torch.permute(
                    torch.tensor(data.observations.data),
                    (2, 0, 1)
                )
                inpt_data = inpt_data.to(dtype=torch.float32)


            with xr.open_dataset(self.target_files[ind]) as data:
                targets = {}
                for name in TARGETS:
                    if name in data:
                        targ = torch.tensor(data[name].data)
                        if targ.ndim > 2:
                            targ = torch.permute(targ, (2, 0, 1))
                        targets[name] = targ.to(torch.float32)
                    else:
                        targ = torch.nan * torch.zeros(TARGETS[name] + (256, 256))
                        targets[name] = targ
        except:
            LOGGER.warning(
                "Encountered an error when loading data from input/target files %s/%s.",
                self.input_files[ind],
                self.target_files[ind]
            )
            new_ind = self.rng.integers(0, len(self))
            return self[new_ind]

        twc = targets.pop("total_water_content")
        targets["snow_water_content"] = twc - targets["rain_water_content"]

        latent_heating = targets["latent_heating"]
        latent_heating[latent_heating < -1000] = torch.nan

        precip_type = targets["precipitation_type"]
        precip_type[precip_type == 0.0] = torch.nan
        precip_type -= 1.0

        convective = torch.nan * torch.ones_like(precip_type)
        convective[precip_type > 0.0] = 0.0
        convective[torch.isclose(precip_type, torch.tensor(2.0))] = 1.0
        targets["convective"] = convective
        conv_precip = np.nan * torch.clone(targets["surface_precip"])
        conv_mask = convective >= 1.0
        conv_precip[conv_mask] = targets["surface_precip"][conv_mask]
        other_mask = convective < 1.0
        conv_precip[other_mask] = 0.0

        targets["convective_precip"] = conv_precip
        targets = {"surface_precip": targets["surface_precip"]}

        # Horizontal flip
        if self.augment:
            if self.rng.random() > 0.5:
                inpt_data = torch.flip(inpt_data, (-1,))
                targets = {
                    name: torch.flip(targ, (-1,)) for name, targ in targets.items()
                }
            if self.rng.random() > 0.5:
                inpt_data = torch.flip(inpt_data, (-2,))
                targets = {
                    name: torch.flip(targ, (-2,)) for name, targ in targets.items()
                }

        return inpt_data, targets
