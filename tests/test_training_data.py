"""
Tests for the noaagprof.data.training_data module.
"""
from datetime import datetime, timedelta

import numpy as np
import pytest
import torch
import xarray as xr

from noaagprof.training_data import TrainingDataset


@pytest.fixture
def training_data_mrms(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("mrms")
    obs_folder = tmp_path / "noaa"
    obs_folder.mkdir()
    target_folder = tmp_path / "target"
    target_folder.mkdir()


    time = datetime(2020, 1, 1)
    while time < datetime(2020, 1, 1, 1):

        fname = f"noaa_{time.strftime('%Y%m%d%H%M%S')}.nc"
        input_data = xr.Dataset({
            "observations": (
               ("scans", "pixels", "channels"),
                np.zeros((256, 256, 16,), dtype="float32")
            )
        })
        input_data.to_netcdf(obs_folder / fname)

        fname = f"target_{time.strftime('%Y%m%d%H%M%S')}.nc"
        target_data = xr.Dataset({
            "surface_precip": (("scans", "pixels",), np.zeros((256, 256), dtype="float32")),
            "radar_quality_index": (("scans", "pixels",), np.zeros((256, 256), dtype="float32")),
            "valid_fraction": (("scans", "pixels",), np.zeros((256, 256), dtype="float32")),
            "hail_fraction": (("scans", "pixels",), np.zeros((256, 256), dtype="float32")),
            "convective_fraction": (("scans", "pixels",), np.zeros((256, 256), dtype="float32")),
            "stratiform_fraction": (("scans", "pixels",), np.zeros((256, 256), dtype="float32")),
        })
        target_data.to_netcdf(target_folder / fname)

        time += timedelta(minutes=15)

    return tmp_path


def test_training_data_mrms(training_data_mrms):
    """
    Ensure that the training dataset successfully identifies and loads training
    data from training files extracted from MRMS collocations.
    """
    training_data = TrainingDataset(training_data_mrms)
    assert len(training_data) == 4

    x, y = training_data[0]

    assert torch.isfinite(y["surface_precip"]).any()
    assert torch.isnan(y["rain_water_content"]).all()
    assert torch.isnan(y["snow_water_content"]).all()


@pytest.fixture
def training_data_cmb(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("cmb")
    obs_folder = tmp_path / "noaa"
    obs_folder.mkdir()
    target_folder = tmp_path / "target"
    target_folder.mkdir()


    time = datetime(2020, 1, 1)
    while time < datetime(2020, 1, 1, 1):

        fname = f"noaa_{time.strftime('%Y%m%d%H%M%S')}.nc"
        input_data = xr.Dataset({
            "observations": (
               ("scans", "pixels", "channels"),
                np.zeros((256, 256, 16,), dtype="float32")
            )
        })
        input_data.to_netcdf(obs_folder / fname)

        fname = f"target_{time.strftime('%Y%m%d%H%M%S')}.nc"
        target_data = xr.Dataset({
            "surface_precip": (("scans", "pixels",), np.zeros((256, 256), dtype="float32")),
            "precipitation_type": (("scans", "pixels",), np.zeros((256, 256), dtype="float32")),
            "total_water_content": (("scans", "pixels", "layers"), np.zeros((256, 256, 28), dtype="float32")),
            "rain_water_content": (("scans", "pixels", "layers"), np.zeros((256, 256, 28), dtype="float32")),
            "rain_water_path": (("scans", "pixels",), np.zeros((256, 256), dtype="float32")),
            "snow_water_path": (("scans", "pixels",), np.zeros((256, 256), dtype="float32")),
            "latent_heating": (("scans", "pixels", "layers"), np.zeros((256, 256, 28), dtype="float32")),
        })
        target_data.to_netcdf(target_folder / fname)

        time += timedelta(minutes=15)

    return tmp_path


def test_training_data_cmb(training_data_cmb):
    """
    Ensure that the training dataset successfully identifies and loads training
    data from training files extracted from CMB collocations.
    """
    training_data = TrainingDataset(training_data_cmb)
    assert len(training_data) == 4

    x, y = training_data[0]

    assert torch.isfinite(y["surface_precip"]).all()
    assert torch.isfinite(y["snow_water_content"]).all()
    assert torch.isfinite(y["snow_water_content"]).all()


def test_training_data_mrms_and_cmb(training_data_mrms, training_data_cmb):
    """
    Ensure that the training dataset successfully identifies and loads training
    data from training files extracted from CMB collocations.
    """
    training_data = TrainingDataset([training_data_mrms, training_data_cmb])
    assert len(training_data) == 8

    x, y = training_data[0]
    assert torch.isfinite(y["surface_precip"]).all()
