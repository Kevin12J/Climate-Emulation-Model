import os
from datetime import datetime

import dask.array as da
import hydra
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xarray as xr
from hydra.utils import to_absolute_path
from lightning.pytorch import LightningDataModule
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from typing import List
from torch.utils.data import ConcatDataset
from src.loss import make_gabor_bank
from src.loss import composite_tas_pr_loss
from src.dataset import ClimateSequenceDataset
import math



try:
    import wandb  # Optional, for logging to Weights & Biases
except ImportError:
    wandb = None

from src.models import get_model
from src.utils import (
    Normalizer,
    calculate_weighted_metric,
    convert_predictions_to_kaggle_format,
    create_climate_data_array,
    create_comparison_plots,
    get_lat_weights,
    get_logger,
    get_trainer_config,
)


# Setup logging
log = get_logger(__name__)


# --- Data Handling ---


# Dataset to precompute all tensors during initialization
class ClimateDataset(Dataset):
    def __init__(self, inputs_norm_dask, outputs_dask, output_is_normalized=True):
        # Store dataset size
        self.size = inputs_norm_dask.shape[0]

        # Log once with basic information
        log.info(
            f"Creating dataset: {self.size} samples, input shape: {inputs_norm_dask.shape}, normalized output: {output_is_normalized}"
        )

        # Precompute all tensors in one go 
        # (arrays are in dask format initially, which is an efficient storage method)
        # they are lazily evaluated (wow haskell pilled???)
        # lazy evaluation, for everyone else means that until they are used 
        # (i.e. call compute() on them, which converts them to numpy arrays),
        # they are either not in memory at all or only a subset is available
        inputs_np = inputs_norm_dask.compute()
        outputs_np = outputs_dask.compute()

        # Convert to PyTorch tensors
        self.input_tensors = torch.from_numpy(inputs_np).float()
        self.output_tensors = torch.from_numpy(outputs_np).float()

        # Handle NaN values (should not occur)
        if torch.isnan(self.input_tensors).any() or torch.isnan(self.output_tensors).any():
            raise ValueError("NaN values detected in dataset tensors")

    # boilerplate pytorch dataset stuff
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.input_tensors[idx], self.output_tensors[idx]
    



# important point; everything up to this point is done in xarrays, not nparrays
# these are good for labeled data beyond just 2 dimensions
def _load_process_ssp_data(ds, ssp, input_variables, output_variables, member_id, spatial_template):
    """
    Loads and processes input and output variables for a single SSP using Dask.

    # SSP means Shared Socioeconomic Pathway
    # this seemed stupid but is probably pretty logical
    # It encompasses economics, demographics, etc...
    # economic developments can include green boons, demographics can be population growth, etc...
    # I imagine the goal is to train over a dataset under one ssp identifier (one socioeconomic context)
    # and see how our predictions are influenced when evaluating on different ones.
    # While ssp may not be a direct input, it influences the whole datasets conditions, and so this makes sense

    # Note: SSPs here are used for splitting based on train/val/test
    # One ssp gets chosen for testing, and two get split into val/train 
    # (370 specifically is the val one; 120 months are taken as val, the rest go into training)
    # Again, they are not explicitly kept when we process the data. At the end, we just have
    # (time, channel (CO2, BHC, feature basically), y, x)



    Args:
        ds (xr.Dataset): The opened xarray dataset.
        ssp (str): The SSP identifier (e.g., 'ssp126').
        input_variables (list): List of input variable names.
        output_variables (list): List of output variable names.
        member_id (int): The member ID to select.
        spatial_template (xr.DataArray): A template DataArray with ('y', 'x') dimensions
                                          for broadcasting global variables.

    Returns:
        tuple: (input_dask_array, output_dask_array)
               - input_dask_array: Stacked dask array of inputs (time, channels, y, x).
               - output_dask_array: Stacked dask array of outputs (time, channels, y, x).
    """
    ssp_input_dasks = []
    # processing step
    for var in input_variables:
        # select based on ssp context
        da_var = ds[var].sel(ssp=ssp)
        # Rename spatial dims if needed (# rename lat and long to y and x)
        if "latitude" in da_var.dims:
            da_var = da_var.rename({"latitude": "y", "longitude": "x"})

        # Select member if applicable 
        # (we run each simulation 3 times; we are reducing it down to one by only getting the inputs where the member id equals our argument)
        if "member_id" in da_var.dims:
            da_var = da_var.sel(member_id=member_id)

        # Process based on dimensions
        # all this code is doing is making sure our data is in the form (time, y, x)
        # something like CO2 has no spatial dimension initially, we process the data here to transform sometihngl ike CO2
        # into a 3d tensor of time, x, y, where each value in the tensor stores the data point
        if set(da_var.dims) == {"time"}:  # Global variable, broadcast to spatial dims:
            # Broadcast like template, then transpose to ensure ('time', 'y', 'x')
            da_var_expanded = da_var.broadcast_like(spatial_template).transpose("time", "y", "x")
            ssp_input_dasks.append(da_var_expanded.data)
        elif set(da_var.dims) == {"time", "y", "x"}:  # Spatially resolved
            ssp_input_dasks.append(da_var.data)
        else:
            raise ValueError(f"Unexpected dimensions for variable {var} in SSP {ssp}: {da_var.dims}")
        
    # this is where we would add something like seasons 
    # (remember we would have to add a longtitude check b/c the seasons are flipped 
    # in the southern and northern hemipsheres, it is NOT just a month check)
    time_months = ds.time.dt.month.values  # shape: [T]
    latitudes = spatial_template["y"].values  # shape: [H]
    longitudes = spatial_template["x"].values  # shape: [W]

    T, H, W = len(time_months), len(latitudes), len(longitudes)
    seasonal_array = np.empty((T, H, W), dtype=np.float32)

    for t in range(T):
        month = time_months[t]
        phase = 2 * np.pi * (month - 1) / 12.0  # Normalize month to [0, 2π]
        lat_scaling = np.cos(np.radians(latitudes))  # strength of seasonality
        hemi_sign = np.sign(latitudes)  # +1 for north, -1 for south
        seasonal_array[t, :, :] = (hemi_sign * lat_scaling)[:, None] * np.cos(phase)

    seasonal_da = xr.DataArray(
        seasonal_array,
        coords={"time": ds.time, "y": spatial_template.y, "x": spatial_template.x},
        dims=("time", "y", "x")
    )

    ssp_input_dasks.append(seasonal_da.data)  # Append to inputs
    # Stack inputs along channel dimension -> dask array (time, channels, y, x)
    # we stack our inputs to get a 4d tensor. E.g. lets say CO2 and CH4 are our input variables
    # these are each 3d tensors of (time * y * x)
    # we stack them to get one tensor
    stacked_input_dask = da.stack(ssp_input_dasks, axis=1)


    # Prepare output dask arrays for each output variable
    # we are preparing out outputs (targets for training) in exactly the same way as the inputs
    output_dasks = []
    for var in output_variables:
        da_output = ds[var].sel(ssp=ssp, member_id=member_id)
        # Ensure output also uses y, x if necessary
        if "latitude" in da_output.dims:
            da_output = da_output.rename({"latitude": "y", "longitude": "x"})

        # Add time, y, x dimensions as a dask array
        output_dasks.append(da_output.data)

    # Stack outputs along channel dimension -> dask array (time, channels, y, x)
    stacked_output_dask = da.stack(output_dasks, axis=1)
    return stacked_input_dask, stacked_output_dask


# data module in lightning is just a data structure holding training/val/test dataloaders
class ClimateEmulationDataModule(LightningDataModule):
    def __init__(
        self,
        path: str,
        input_vars: list,
        output_vars: list,
        train_ssps: list,
        test_ssp: str,
        target_member_id: List[int],
        seq_length: int = 12,
        test_months: int = 360,
        batch_size: int = 32,
        eval_batch_size: int = None,
        num_workers: int = 0,
        seed: int = 42,
    ):
        super().__init__()

        # with save_hyperparameters(), pytorch lightning basically just takes everything from the args
        # and stores them in hparams, which can just be accessed at runtime for debugging info
        self.save_hyperparameters()
        self.hparams.path = to_absolute_path(path) # we reset it to make sure lightning has the absolute path, not the relative path
        self.normalizer = Normalizer()

        # Set evaluation batch size to training batch size if not specified
        if eval_batch_size is None:
            self.hparams.eval_batch_size = batch_size

        # Placeholders
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        self.lat_coords, self.lon_coords, self._lat_weights_da = None, None, None

        self.seq_length = seq_length

    def prepare_data(self):
        if not os.path.exists(self.hparams.path):
            raise FileNotFoundError(f"Data path not found: {self.hparams.path}")
        log.info(f"Data found at: {self.hparams.path}")

    def setup(self, stage: str | None = None):
        log.info(f"Setting up data module for stage: {stage} from {self.hparams.path}")
        train_datasets = []
        # Use context manager for opening dataset
        with xr.open_zarr(self.hparams.path, consolidated=True, chunks={"time": 24}) as ds:
            # Create a spatial template ONCE using a variable guaranteed to have y, x
            # Extract the template DataArray before renaming for coordinate access
            spatial_template_da = ds["rsdt"].isel(time=0, ssp=0, drop=True)  # drop time/ssp dims

            # --- Prepare Training and Validation Data ---
            train_inputs_dask_list, train_outputs_dask_list = [], []
            val_input_dask, val_output_dask = None, None
            val_ssp = "ssp370"
            val_months = 120

            # Process all SSPs
            log.info(f"Loading data from SSPs: {self.hparams.train_ssps}")
            for ssp in self.hparams.train_ssps:
                for member_id in self.hparams.target_member_id:
                    # Load the data for this SSP
                    ssp_input_dask, ssp_output_dask = _load_process_ssp_data(
                        ds,
                        ssp,
                        self.hparams.input_vars,
                        self.hparams.output_vars,
                        member_id,
                        spatial_template_da,
                    )

                    # val split codeh appens here; interesting that we only take the last 120 months for validation
                    if ssp == val_ssp and member_id==0:
                        # Special handling for SSP 370: split into training and validation
                        # Last 120 months go to validation
                        val_input_dask = ssp_input_dask[-val_months:]
                        val_output_dask = ssp_output_dask[-val_months:]
                        # Early months go to training if there are any
                        train_inputs_dask_list.append(ssp_input_dask[:-val_months])
                        train_outputs_dask_list.append(ssp_output_dask[:-val_months])
                    else:
                        # All other SSPs go entirely to training
                        train_inputs_dask_list.append(ssp_input_dask)
                        train_outputs_dask_list.append(ssp_output_dask)


            # normalization steps; basically does batch normalization (mean/var based) on the inputs here

            # Concatenate training data only
            train_input_dask = da.concatenate(train_inputs_dask_list, axis=0)
            train_output_dask = da.concatenate(train_outputs_dask_list, axis=0)

            # # Compute z-score normalization statistics using the training data
            input_mean = da.nanmean(train_input_dask, axis=(0, 2, 3), keepdims=True).compute()
            input_std = da.nanstd(train_input_dask, axis=(0, 2, 3), keepdims=True).compute()
            output_mean = da.nanmean(train_output_dask, axis=(0, 2, 3), keepdims=True).compute()
            output_std = da.nanstd(train_output_dask, axis=(0, 2, 3), keepdims=True).compute()

            self.normalizer.set_input_statistics(mean=input_mean, std=input_std)
            self.normalizer.set_output_statistics(mean=output_mean, std=output_std)

            # # --- Define Normalized Training Dask Arrays ---
            # Added to make sure we don't cross data across member ids
            for ssp in self.hparams.train_ssps:
                for member_id in self.hparams.target_member_id:
                    # Load the data for this SSP
                    ssp_input_dask, ssp_output_dask = _load_process_ssp_data(
                        ds,
                        ssp,
                        self.hparams.input_vars,
                        self.hparams.output_vars,
                        member_id,
                        spatial_template_da,
                    )
                    train_inputs_dask_list = []
                    train_outputs_dask_list = []
                    # val split codeh appens here; interesting that we only take the last 120 months for validation
                    if ssp == val_ssp and member_id==0:
                        # Special handling for SSP 370: split into training and validation
                        # Last 120 months go to validation
                        val_input_dask = ssp_input_dask[-val_months:]
                        val_output_dask = ssp_output_dask[-val_months:]
                        # Early months go to training if there are any
                        train_inputs_dask_list.append(ssp_input_dask[:-val_months])
                        train_outputs_dask_list.append(ssp_output_dask[:-val_months])
                    else:
                        # All other SSPs go entirely to training
                        train_inputs_dask_list.append(ssp_input_dask)
                        train_outputs_dask_list.append(ssp_output_dask)
                    train_input_dask = da.concatenate(train_inputs_dask_list, axis=0)
                    train_output_dask = da.concatenate(train_outputs_dask_list, axis=0)
                    train_input_norm_dask = self.normalizer.normalize(train_input_dask, data_type="input")
                    train_output_norm_dask = self.normalizer.normalize(train_output_dask, data_type="output")
                    train_datasets.append(ClimateSequenceDataset(train_input_norm_dask, train_output_norm_dask,seq_len=self.seq_length,output_is_normalized=True))

            # --- Define Normalized Validation Dask Arrays ---
            val_input_norm_dask = self.normalizer.normalize(val_input_dask, data_type="input")
            val_output_norm_dask = self.normalizer.normalize(val_output_dask, data_type="output")
            
            # --- Prepare Test Data ---
            full_test_input_dask, full_test_output_dask = _load_process_ssp_data(
                ds,
                self.hparams.test_ssp,
                self.hparams.input_vars,
                self.hparams.output_vars,
                0,
                spatial_template_da,
            )

            # --- Slice Test Data ---
            test_slice = slice(-self.hparams.test_months-self.seq_length, None)

            sliced_test_input_dask = full_test_input_dask[test_slice]
            sliced_test_output_raw_dask = full_test_output_dask[test_slice]

            # --- Define Normalized Test Input Dask Array ---
            test_input_norm_dask = self.normalizer.normalize(sliced_test_input_dask, data_type="input")
            test_output_raw_dask = sliced_test_output_raw_dask  # Keep unnormed for evaluation

        # Create datasets
        self.train_dataset = ConcatDataset(train_datasets)
        self.val_dataset = ClimateSequenceDataset(val_input_norm_dask, val_output_norm_dask,seq_len=self.seq_length,output_is_normalized=True)
        self.test_dataset = ClimateSequenceDataset(test_input_norm_dask, test_output_raw_dask,seq_len=self.seq_length,output_is_normalized=False)

        # Log dataset sizes in a single message
        log.info(
            f"Datasets created. Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)} (last months of {val_ssp}), Test: {len(self.test_dataset)}"
        )

    # Common DataLoader configuration
    def _get_dataloader_kwargs(self, is_train=False):
        """Return common DataLoader configuration as a dictionary"""
        return {
            "batch_size": self.hparams.batch_size if is_train else self.hparams.eval_batch_size,
            "shuffle": is_train,  # Only shuffle training data
            "num_workers": self.hparams.num_workers,
            "persistent_workers": self.hparams.num_workers > 0,
            "pin_memory": True,
        }

    # boilerplate dataloader stuff
    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self._get_dataloader_kwargs(is_train=True))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self._get_dataloader_kwargs(is_train=False))

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self._get_dataloader_kwargs(is_train=False))

    # area weights correct for the fact that some long/lat areas are literally smaller in area
    # as unfortunately we do not live on a flat planet
    # this is useful for averaging values
    def get_lat_weights(self):
        """
        Returns area weights for the latitude dimension as an xarray DataArray.
        The weights can be used with xarray's weighted method for proper spatial averaging.
        """
        
        if self._lat_weights_da is None:
            with xr.open_zarr(self.hparams.path, consolidated=True) as ds:
                template = ds["rsdt"].isel(time=0, ssp=0)
                y_coords = template.y.values

                # Calculate weights based on cosine of latitude
                weights = get_lat_weights(y_coords)

                # Create DataArray with proper dimensions
                self._lat_weights_da = xr.DataArray(weights, dims=["y"], coords={"y": y_coords}, name="area_weights")

        return self._lat_weights_da

    # helper method that gets all coord pairs. Useful for building plots
    def get_coords(self):
        """
        Returns the y and x coordinates (representing latitude and longitude).

        Returns:
            tuple: (y array, x array)
        """
        if self.lat_coords is None or self.lon_coords is None:
            # Get coordinates if they haven't been stored yet
            with xr.open_zarr(self.hparams.path, consolidated=True) as ds:
                template = ds["rsdt"].isel(time=0, ssp=0, drop=True)
                self.lat_coords = template.y.values
                self.lon_coords = template.x.values

        return self.lat_coords, self.lon_coords



# the rest of this makes sense to me. Feel free to add comments if any of you are confused
# --- PyTorch Lightning Module ---
class ClimateEmulationModule(pl.LightningModule):
    def __init__(self, model: nn.Module, learning_rate: float,
                kernel_size_gabor: int = 7,
                gabor_sigmas: list[float] = [1.0, 2.0],
                gabor_lambdas: list[float] = [3.0, 5.0],
                gabor_thetas: list[float] = [i * math.pi / 8 for i in range(8)]):
        super().__init__()
        self.model = model
        # Access hyperparams via self.hparams object after saving, e.g., self.hparams.learning_rate
        self.save_hyperparameters(ignore=["model"])
        self.criterionCustom = composite_tas_pr_loss
        self.latWeights = None
        self.criterion = nn.MSELoss()
        self.normalizer = None
        # Store evaluation outputs for time-mean calculation
        self.test_step_outputs = []
        self.validation_step_outputs = []

        gabor_kernels = make_gabor_bank(
            kernel_size=kernel_size_gabor,
            sigmas=gabor_sigmas,
            lambdas=gabor_lambdas,
            thetas=gabor_thetas,
        )
        # Register so Lightning moves it to GPU automatically.
        self.register_buffer("gabor_kernels", gabor_kernels, persistent=False)

        dxx = torch.tensor(
            [[0.0, 0.0, 0.0],
             [1.0, -2.0, 1.0],
             [0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        # ∂²/∂y²: [[0,1,0],[0,-2,0],[0,1,0]]
        dyy = torch.tensor(
            [[0.0, 1.0, 0.0],
             [0.0, -2.0, 0.0],
             [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        # ∂²/∂x∂y (central‐difference 3×3): [[1,0,-1],[0,0,0],[-1,0,1]] / 4
        # We omit the “/4” normalization since we're only comparing pred vs true
        # and taking an L2 on differences. Multiplying all entries by 1/4
        # would just scale the loss by a constant.
        dxy = torch.tensor(
            [[ 1.0, 0.0, -1.0],
             [ 0.0, 0.0,  0.0],
             [-1.0, 0.0,  1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        self.register_buffer("dxx_k", dxx, persistent=False)
        self.register_buffer("dyy_k", dyy, persistent=False)
        self.register_buffer("dxy_k", dxy, persistent=False)

        self.center_x = None
        self.center_y = None

    def forward(self, x):
        return self.model(x)

    def on_fit_start(self) -> None:
        self.normalizer = self.trainer.datamodule.normalizer  # Access the normalizer from the datamodule

    def training_step(self, batch, batch_idx):
        x, y_true_norm= batch
        y_pred_norm = self(x)
        if self.latWeights is None:
            self.latWeights = torch.tensor(np.tile(np.expand_dims(self.trainer.datamodule.get_lat_weights().values,axis=1),(1,72)),device=y_pred_norm.device)
        loss = self.criterionCustom(y_pred_norm,y_true_norm,self.trainer.datamodule.hparams.output_vars.index("tas"),self.trainer.datamodule.hparams.output_vars.index("pr"),self)
        self.log("train/loss", loss, prog_bar=True, batch_size=x.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true_norm = batch
        y_pred_norm = self(x)
        loss = self.criterion(y_pred_norm, y_true_norm)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.size(0), sync_dist=True)

        # Save unnormalized outputs for decadal mean/stddev calculation in validation_epoch_end
        y_pred_norm = self.normalizer.inverse_transform_output(y_pred_norm.cpu().numpy())
        y_true_norm = self.normalizer.inverse_transform_output(y_true_norm.cpu().numpy())

        self.validation_step_outputs.append((y_pred_norm, y_true_norm))

        return loss

    def _evaluate_predictions(self, predictions, targets, is_test=False):
        """
        Helper method to evaluate predictions against targets with climate metrics.

        Args:
            predictions (np.ndarray): Prediction array with shape (time, channels, y, x)
            targets (np.ndarray): Target array with shape (time, channels, y, x)
            is_test (bool): Whether this is being called from test phase (vs validation)
        """
        phase = "test" if is_test else "val"
        log_kwargs = {"prog_bar": not is_test, "sync_dist": not is_test}

        # Get number of evaluation timesteps
        n_timesteps = predictions.shape[0]

        # Get area weights for proper spatial averaging
        area_weights = self.trainer.datamodule.get_lat_weights()

        # Get coordinates
        lat_coords, lon_coords = self.trainer.datamodule.get_coords()
        time_coords = np.arange(n_timesteps)
        output_vars = self.trainer.datamodule.hparams.output_vars

        # Process each output variable
        for i, var_name in enumerate(output_vars):
            # Extract channel data
            preds_var = predictions[:, i, :, :]
            trues_var = targets[:, i, :, :]

            var_unit = "mm/day" if var_name == "pr" else "K" if var_name == "tas" else "unknown"

            # Create xarray objects for weighted calculations
            preds_xr = create_climate_data_array(
                preds_var, time_coords, lat_coords, lon_coords, var_name=var_name, var_unit=var_unit
            )
            trues_xr = create_climate_data_array(
                trues_var, time_coords, lat_coords, lon_coords, var_name=var_name, var_unit=var_unit
            )

            # 1. Calculate weighted month-by-month RMSE over all samples
            diff_squared = (preds_xr - trues_xr) ** 2
            overall_rmse = calculate_weighted_metric(diff_squared, area_weights, ("time", "y", "x"), "rmse")
            self.log(f"{phase}/{var_name}/avg/monthly_rmse", float(overall_rmse), **log_kwargs)

            # 2. Calculate time-mean (i.e. decadal, 120 months average) and calculate area-weighted RMSE for time means
            pred_time_mean = preds_xr.mean(dim="time")
            true_time_mean = trues_xr.mean(dim="time")
            mean_diff_squared = (pred_time_mean - true_time_mean) ** 2
            time_mean_rmse = calculate_weighted_metric(mean_diff_squared, area_weights, ("y", "x"), "rmse")
            self.log(f"{phase}/{var_name}/time_mean_rmse", float(time_mean_rmse), **log_kwargs)

            # 3. Calculate time-stddev (temporal variability) and calculate area-weighted MAE for time stddevs
            pred_time_std = preds_xr.std(dim="time")
            true_time_std = trues_xr.std(dim="time")
            std_abs_diff = np.abs(pred_time_std - true_time_std)
            time_std_mae = calculate_weighted_metric(std_abs_diff, area_weights, ("y", "x"), "mae")
            self.log(f"{phase}/{var_name}/time_stddev_mae", float(time_std_mae), **log_kwargs)

            # Extra logging of sample predictions/images to wandb for test phase (feel free to use this for validation)
            if is_test:
                # Generate visualizations for test phase when using wandb
                if isinstance(self.logger, WandbLogger):
                    # Time mean visualization
                    fig = create_comparison_plots(
                        true_time_mean,
                        pred_time_mean,
                        title_prefix=f"{var_name} Mean",
                        metric_value=time_mean_rmse,
                        metric_name="Weighted RMSE",
                    )
                    self.logger.experiment.log({f"img/{var_name}/time_mean": wandb.Image(fig)})
                    plt.close(fig)

                    # Time standard deviation visualization
                    fig = create_comparison_plots(
                        true_time_std,
                        pred_time_std,
                        title_prefix=f"{var_name} Stddev",
                        metric_value=time_std_mae,
                        metric_name="Weighted MAE",
                        cmap="plasma",
                    )
                    self.logger.experiment.log({f"img/{var_name}/time_Stddev": wandb.Image(fig)})
                    plt.close(fig)

                    # Sample timesteps visualization
                    if n_timesteps > 3:
                        timesteps = np.random.choice(n_timesteps, 3, replace=False)
                        for t in timesteps:
                            true_t = trues_xr.isel(time=t)
                            pred_t = preds_xr.isel(time=t)
                            fig = create_comparison_plots(true_t, pred_t, title_prefix=f"{var_name} Timestep {t}")
                            self.logger.experiment.log({f"img/{var_name}/month_idx_{t}": wandb.Image(fig)})
                            plt.close(fig)

    def on_validation_epoch_end(self):
        # Compute time-mean and time-stddev errors using all validation months
        if not self.validation_step_outputs:
            return

        # Stack all predictions and ground truths
        all_preds_np = np.concatenate([pred for pred, _ in self.validation_step_outputs], axis=0)
        all_trues_np = np.concatenate([true for _, true in self.validation_step_outputs], axis=0)

        # Use the helper method to evaluate predictions
        self._evaluate_predictions(all_preds_np, all_trues_np, is_test=False)

        self.validation_step_outputs.clear()  # Clear the outputs list for next epoch

    def test_step(self, batch, batch_idx):
        x, y_true_denorm = batch
        y_pred_norm = self(x)
        # Denormalize the predictions for evaluation back to original scale
        y_pred_denorm = self.normalizer.inverse_transform_output(y_pred_norm.cpu().numpy())

        # reverse log
        pr_index = self.trainer.datamodule.hparams.output_vars.index("pr") 
        #y_pred_denorm[:, pr_index, :, :] = np.expm1(y_pred_denorm[:, pr_index, :, :])
        #y_pred_denorm[:, pr_index, :, :] = np.square(y_pred_denorm[:, pr_index, :, :])
        y_true_denorm_np = y_true_denorm.cpu().numpy()
        self.test_step_outputs.append((y_pred_denorm, y_true_denorm_np))

    def on_test_epoch_end(self):
        # Concatenate all predictions and ground truths from each test step/batch into one array
        all_preds_denorm = np.concatenate([pred for pred, true in self.test_step_outputs], axis=0)
        all_trues_denorm = np.concatenate([true for pred, true in self.test_step_outputs], axis=0)

        # Use the helper method to evaluate predictions
        self._evaluate_predictions(all_preds_denorm, all_trues_denorm, is_test=True)

        # Save predictions for Kaggle submission. This is the file that should be uploaded to Kaggle.
        log.info("Saving Kaggle submission...")
        self._save_kaggle_submission(all_preds_denorm)

        self.test_step_outputs.clear()  # Clear the outputs list

    


    def _save_kaggle_submission(self, predictions, suffix=""):
        """
        Create a Kaggle submission file from the model predictions.

        Args:
            predictions (np.ndarray): Predicted values with shape (time, channels, y, x)
        """
        # Get coordinates
        lat_coords, lon_coords = self.trainer.datamodule.get_coords()
        output_vars = self.trainer.datamodule.hparams.output_vars
        n_times = predictions.shape[0]
        time_coords = np.arange(n_times)

        # Convert predictions to Kaggle format
        submission_df = convert_predictions_to_kaggle_format(
            predictions, time_coords, lat_coords, lon_coords, output_vars
        )

        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = to_absolute_path(f"submissions/kaggle_submission{suffix}_{timestamp}.csv")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Ensure directory exists
        submission_df.to_csv(filepath, index=False)

        if wandb is not None and isinstance(self.logger, WandbLogger):
            #pass
            # Optionally, uncomment the following line to save the submission to the wandb cloud
            self.logger.experiment.log_artifact(filepath)  # Log to wandb if available

        log.info(f"Kaggle submission saved to {filepath}")

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate,weight_decay=1e-2)
        #optimizer = optim.RAdam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

        return {
            'optimizer': optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                #"monitor": "val/loss",  # ← this must match what you're logging
                #"interval": "epoch",
                #"frequency": 1
            }
        }


# --- Main Execution with Hydra ---
@hydra.main(version_base=None, config_path="configs", config_name="main_config.yaml")
def main(cfg: DictConfig):
    # Print resolved configs
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Set random seed
    pl.seed_everything(cfg.seed, workers=True)

    # Create data module with parameters from configs
    datamodule = ClimateEmulationDataModule(seed=cfg.seed, **cfg.data)
    model = get_model(cfg)

    # Create lightning module
    lightning_module = ClimateEmulationModule(model, learning_rate=cfg.training.lr)
    # Create lightning trainer
    trainer_config = get_trainer_config(cfg, model=model)
    trainer = pl.Trainer(**trainer_config)

    if cfg.ckpt_path and isinstance(cfg.ckpt_path, str):
        cfg.ckpt_path = to_absolute_path(cfg.ckpt_path)

    # Train model
    trainer.fit(lightning_module, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    log.info("Training finished.")

    # Test model
    # IMPORTANT: Please note that the test metrics will be bad because the test targets have been corrupted on the public Kaggle dataset.
    # The purpose of testing below is to generate the Kaggle submission file based on your model's predictions.
    trainer_config["devices"] = 1  # Make sure you test on 1 GPU only to avoid synchronization issues with DDP
    eval_trainer = pl.Trainer(**trainer_config)
    eval_trainer.test(lightning_module, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    if cfg.use_wandb and isinstance(trainer_config.get("logger"), WandbLogger):
        wandb.finish()  # Finish the run if using wandb


if __name__ == "__main__":
    main()
