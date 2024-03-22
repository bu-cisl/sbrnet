import datetime
import logging
import os
from pyexpat import model
from typing import Tuple
import time
from pandas import read_parquet
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda.amp import GradScaler, autocast
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset

from sbrnet_core.sbrnet.dataset import CustomDataset, PatchDataset
from sbrnet_core.sbrnet.models.noisemodel import PoissonGaussianNoiseModel
from sbrnet_core.sbrnet.losses.quantile_loss import QuantileLoss
from sbrnet_core.sbrnet.calibration.rcps import get_preds

from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt


now = datetime.datetime.now()

timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

## for debugging
# torch.cuda.empty_cache()
# import os

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"
##


logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: Module,
        config: dict,
    ):
        self.config = config
        self.model = model
        self.noise_model = PoissonGaussianNoiseModel(config)
        self.learning_rate = config["learning_rate"]
        self.epochs = config["epochs"]
        self.model_dir = config["model_dir"]
        self.lowest_val_loss = float("inf")
        self.training_losses = []
        self.qlo_training_losses = []
        self.qhi_training_losses = []
        self.validation_losses = []
        self.qlo_validation_losses = []
        self.qhi_validation_losses = []
        self.random_seed = config.get("random_seed", None)
        self.use_amp = config.get("use_amp", False)
        self.optimizer_name = config.get("optimizer", "adam")
        self.lr_scheduler_name = config.get("lr_scheduler", "cosine_annealing")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        logger.debug(f"Using device: {self.device}")
        self.scaler = (
            GradScaler() if self.use_amp else None
        )  # Initialize the scaler if using AMP

        # Initialize the loss criterion based on the config
        if config["last_layer"] == "quantile_heads":
            self.criterion = QuantileLoss(config)
        else:
            if config["criterion_name"] == "bce_with_logits":
                self.criterion = nn.BCEWithLogitsLoss()
            elif config["criterion_name"] == "mse":
                self.criterion = nn.MSELoss()
            elif config["criterion_name"] == "mae":
                self.criterion = nn.L1Loss()
            else:
                print(
                    f"Unknown loss criterion: {config['criterion_name']}. Using BCEWithLogitsLoss."
                )
                self.criterion = nn.BCEWithLogitsLoss()
        self.train_data_loader, self.val_data_loader = self._get_dataloaders()
        logger.info(f"Loss criterion: {self.criterion}")

    def _get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        def split_dataset(dataset, split_ratio):
            dataset_size = len(dataset)
            train_size = int(split_ratio * dataset_size)
            val_size = dataset_size - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
            return train_dataset, val_dataset

        complete_dataset: Dataset = CustomDataset(self.config)
        split_ratio = self.config["train_split"]
        train_dataset, val_dataset = split_dataset(complete_dataset, split_ratio)

        # only train_dataset is a PatchDataset. val_dataset is full sized images.
        train_dataset = PatchDataset(
            dataset=train_dataset,
            config=self.config,
        )

        train_dataloader = DataLoader(
            train_dataset,
            self.config.get("batch_size"),
            shuffle=True,
            num_workers=3,
            pin_memory=True,
        )
        val_dataloader = DataLoader(
            val_dataset,
            self.config["batch_size"],
            shuffle=True,
            num_workers=3,
            pin_memory=True,
        )

        return train_dataloader, val_dataloader

    def _set_random_seed(self):
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_seed)

    def _initialize_optimizer(self):
        if self.optimizer_name == "adam":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.config.get("weight_decay", 0.0001),
            )
        elif self.optimizer_name == "sgd":
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.5,
                weight_decay=self.config.get("weight_decay", 0.0001),
            )
        else:
            print(f"Unknown optimizer: {self.optimizer_name}. Using Adam.")
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def _initialize_lr_scheduler(self, optimizer):
        if self.lr_scheduler_name == "cosine_annealing":
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config["cosine_annealing_T_max"]
            )
        elif self.lr_scheduler_name == "cosine_annealing_with_warm_restarts":
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=self.config["cosine_annealing_T_max"]
            )
        elif self.lr_scheduler_name == "step_lr":
            # StepLR scheduler
            step_size = self.config.get("step_lr_step_size", 10)
            gamma = self.config.get("step_lr_gamma", 0.5)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif self.lr_scheduler_name == "plateau":
            # Plateau scheduler
            patience = self.config.get("plateau_patience", 10)
            factor = self.config.get("plateau_factor", 0.5)
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=patience, factor=factor, verbose=True
            )
        else:
            print(
                f"Unknown LR scheduler: {self.lr_scheduler_name}. Using Cosine Annealing."
            )
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        return scheduler

    def train(self):
        logger.info(f"View index: {self.config['view_ind']}")
        model_name = f"sbrnet_view_{self.config['view_ind']}_v0.pt"
        model_save_path = os.path.join(self.model_dir, model_name)

        # Check if the path already exists and make a new version path to save
        while os.path.exists(model_save_path):
            base_name, ext = os.path.splitext(model_name)
            version_str = base_name.split("_")[-1]  # Extract the version string
            if version_str.startswith("v") and version_str[1:].isdigit():
                version = int(version_str[1:])

            version += 1

            model_name = f"sbrnet_view_{self.config['view_ind']}_v{version}"
            model_save_path = os.path.join(self.model_dir, model_name + ".pt")
        logger.info(f"Model save path: {model_save_path}")
        torch.save(self.config, model_save_path)
        self.model.to(self.device)
        self.noise_model.to(self.device)
        self._set_random_seed()

        optimizer = self._initialize_optimizer()
        scheduler = self._initialize_lr_scheduler(optimizer)
        start_time = time.time()

        if self.use_amp:
            logger.info("Using mixed-precision training with AMP.")

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            total_qlo_loss = 0
            total_qhi_loss = 0
            for lf_view_stack, rfv, gt in self.train_data_loader:
                lf_view_stack, rfv, gt = (
                    lf_view_stack.to(self.device),
                    rfv.to(self.device),
                    gt.to(self.device),
                )

                lf_view_stack, rfv = self.noise_model(lf_view_stack, rfv)

                optimizer.zero_grad()

                if self.use_amp:
                    with autocast(enabled=True):
                        output = self.model(lf_view_stack, rfv)
                        loss = self.criterion(output, gt)
                    if isinstance(loss, tuple):
                        loss_all = loss[0]
                        self.scaler.scale(loss_all).backward()
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(optimizer)
                        self.scaler.update()
                else:
                    output = self.model(lf_view_stack, rfv)
                    loss = self.criterion(output, gt)
                    if isinstance(loss, tuple):
                        loss_all = loss[0]
                        loss_all.backward()
                    else:
                        loss.backward()
                    optimizer.step()

                total_loss += loss_all.item()
                total_qlo_loss += loss[1].item() if isinstance(loss, tuple) else 0
                total_qhi_loss += loss[2].item() if isinstance(loss, tuple) else 0

            avg_train_loss = total_loss / len(self.train_data_loader)
            avg_qlo_loss = (
                total_qlo_loss / len(self.train_data_loader)
                if isinstance(loss, tuple)
                else 0
            )
            avg_qhi_loss = (
                total_qhi_loss / len(self.train_data_loader)
                if isinstance(loss, tuple)
                else 0
            )
            self.training_losses.append(avg_train_loss)
            self.qlo_training_losses.append(avg_qlo_loss)
            self.qhi_training_losses.append(avg_qhi_loss)

            logger.info(
                f"Epoch [{epoch + 1}/{self.epochs}], Train Loss: {avg_train_loss}"
            )
            logger.info(f"Time elapsed: {time.time() - start_time} seconds")

            val_loss, val_qlo_loss, val_qhi_loss = self.validate()

            self.validation_losses.append(val_loss)
            self.qlo_validation_losses.append(val_qlo_loss)
            self.qhi_validation_losses.append(val_qhi_loss)

            logger.info(
                f"Epoch [{epoch + 1}/{self.epochs}], Validation Loss: {val_loss}"
            )
            logger.info(f"Time elapsed: {time.time() - start_time} seconds")

            if self.lr_scheduler_name == "plateau":
                scheduler.step(
                    val_loss
                )  # For Plateau scheduler, pass validation loss as an argument
            else:
                scheduler.step()

            if val_loss < self.lowest_val_loss:
                self.lowest_val_loss = val_loss
                save_state = {
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "training_losses": self.training_losses,
                    "validation_losses": self.validation_losses,
                    "q_lo_training_loss": self.qlo_training_losses,
                    "q_lo_validation_loss": self.qlo_validation_losses,
                    "q_hi_training_loss": self.qhi_training_losses,
                    "q_hi_validation_loss": self.qhi_validation_losses,
                    "time_elapsed": time.time() - start_time,
                }

                save_state.update(self.config)

                torch.save(save_state, model_save_path)
                logger.info("Model saved at epoch {}".format(epoch + 1))

                # rough draft for saving a visualization of the model prediction. could be better.
                cal_data = cal_data = CustomDataset(self.config)
                stack, rfv, gt = cal_data[0]
                qlo, qhi, f = get_preds(
                    self.model,
                    stack[None, :, :, :].to(self.device),
                    rfv[None, :, :, :].to(self.device),
                    self.config,
                )
                qlo, qhi, f = (
                    qlo.cpu().numpy().squeeze(),
                    qhi.cpu().numpy().squeeze(),
                    f.cpu().numpy().squeeze(),
                )

                fig, axs = subplots(3, 1, figsize=(15, 15))
                axs = axs.ravel()
                q = 1
                # im0 = axs[0].imshow(np.max(gt, axis=q))
                # axs[0].set_title("Ground Truth")
                # fig.colorbar(im0, ax=axs[0])

                im1 = axs[0].imshow(np.max(qlo, axis=q))
                axs[0].set_title("lower quantile")
                fig.colorbar(im1, ax=axs[0])

                im2 = axs[2].imshow(np.max(qhi, axis=q))
                axs[2].set_title("upper quantile")
                fig.colorbar(im2, ax=axs[2])

                im3 = axs[1].imshow(np.max(f, axis=q))
                axs[1].set_title("point pred")
                fig.suptitle(f"epoch {epoch}")
                fig.colorbar(im3, ax=axs[1])
                fig_save_fol = (
                    "/projectnb/tianlabdl/rjbona/training_visualization/"
                )
                fig_save_path = os.path.join(
                    fig_save_fol, f"{model_name}.png"
                )
                fig.savefig(fig_save_path)
                plt.close(fig)
                

    def validate(self):
        self.model.eval()
        total_loss = 0
        total_qlo_loss = 0
        total_qhi_loss = 0
        with torch.no_grad():
            for lf_view_stack, rfv, gt in self.val_data_loader:
                lf_view_stack, rfv, gt = (
                    lf_view_stack.to(self.device),
                    rfv.to(self.device),
                    gt.to(self.device),
                )
                with autocast(enabled=self.use_amp):
                    lf_view_stack, rfv = self.noise_model(lf_view_stack, rfv)
                    output = self.model(lf_view_stack, rfv)
                    loss = self.criterion(output, gt)
                if isinstance(loss, tuple):
                    loss_all = loss[0]
                    total_loss += loss_all.item()
                else:
                    total_loss += loss.item()
                total_qlo_loss += loss[1].item() if isinstance(loss, tuple) else 0
                total_qhi_loss += loss[2].item() if isinstance(loss, tuple) else 0
        return (
            total_loss / len(self.val_data_loader),
            total_qlo_loss / len(self.val_data_loader),
            total_qhi_loss / len(self.val_data_loader),
        )
