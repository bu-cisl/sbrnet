# Importing modules
import datetime
import logging
import os
from typing import Tuple
import time
from pandas import read_parquet

# Importing torch modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda.amp import GradScaler, autocast
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset

# Importing classes
from sbrnet_core.sbrnet.dataset import CustomDataset, PatchDataset
from sbrnet_core.sbrnet.noisemodel import PoissonGaussianNoiseModel

# Creating timestamp with current time
now = datetime.datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

## For debugging
# torch.cuda.empty_cache()
# import os

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"
##

logger = logging.getLogger(__name__)  # Initializing logger


# Creating Trainer class
class Trainer:
    """
    Trainer class for training a PyTorch Module. The trainer handles setting up the data loaders, optimizer, 
    loss function, learning rate scheduler, and executing the training and validation loop.
    """
    def __init__(self, model: Module, config: dict):
        """
        Initializes the trainer with the given model and configuration settings.

        Args:
            model (Module): The PyTorch module to be trained.
            config (dict): A configuration dictionary containing training parameters.
        """
        # Initialization of internal variables and training setup from the configuration
        self.config = config
        self.model = model
        self.noise_model = PoissonGaussianNoiseModel(config)
        self.learning_rate = config["learning_rate"]
        self.epochs = config["epochs"]
        self.model_dir = config["model_dir"]
        self.lowest_val_loss = float("inf")
        self.training_losses = []
        self.validation_losses = []
        self.random_seed = config.get("random_seed", None)
        self.use_amp = config.get("use_amp", False)
        self.optimizer_name = config.get("optimizer", "adam")
        self.lr_scheduler_name = config.get("lr_scheduler", "cosine_annealing")
        self.criterion_name = config.get("loss_criterion", "bce_with_logits")
        self.architecture = config["backbone"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")  # Log the device being used

        # Create the model directory if it doesn't exist
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Initialize the scaler if using AMP
        self.scaler = (GradScaler() if self.use_amp else None)  

        # Initialize the loss criterion based on the configuration
        if self.criterion_name == "bce_with_logits":
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.criterion_name == "mse":
            self.criterion = nn.MSELoss()
        elif self.criterion_name == "mae":
            self.criterion = nn.L1Loss()
        else:
            print(f"Unknown loss criterion: {self.criterion_name}. Using BCEWithLogitsLoss.")

        # Setup data loaders for training and validation datasets
        self.train_data_loader, self.val_data_loader = self._get_dataloaders()


    def _get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Creates and returns data loaders for the training and validation datasets.

        Returns:
            Tuple[DataLoader, DataLoader]: A tuple containing the training and validation data loaders.
        """
        def split_dataset(dataset, split_ratio):
            """Splits the dataset into training and validation datasets using the given split ratio."""
            dataset_size = len(dataset)
            train_size = int(split_ratio * dataset_size)
            val_size = dataset_size - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            return train_dataset, val_dataset

        # Load complete dataset and split into into training and validation sets
        complete_dataset: Dataset = CustomDataset(self.config["dataset_pq"])
        split_ratio = self.config["train_split"]  # Split ratio determined by config settings
        train_dataset, val_dataset = split_dataset(complete_dataset, split_ratio)

        # Apply patch dataset processing only to the training dataset
        train_dataset = PatchDataset(
            dataset=train_dataset,
            df_path=self.config["dataset_pq"],
            patch_size=self.config["patch_size"],
        )

        # Create data loaders for both training and validation datasets
        train_dataloader = DataLoader(
            train_dataset, self.config.get("batch_size"), shuffle=True
        )
        val_dataloader = DataLoader(
            val_dataset, self.config["batch_size"], shuffle=True
        )

        return train_dataloader, val_dataloader


    def _set_random_seed(self):
        """Sets the random seed for reproducibility if specified in configuration file."""
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)  # Set random seed for CPU
            # If CUDA is available, set the seed for all GPUs
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_seed)


    def _initialize_optimizer(self):
        """Initializes the optimizer based on configuration file."""
        if self.optimizer_name == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == "sgd":
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        else:
            print(f"Unknown optimizer: {self.optimizer_name}. Using Adam.")
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer


    def _initialize_lr_scheduler(self, optimizer):
        """Initializes the learning rate scheduler based on the configuration."""
        if self.lr_scheduler_name == "cosine_annealing":
            # Cosine Annealing Scheduler
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config["cosine_annealing_T_max"])
        elif self.lr_scheduler_name == "step_lr":
            # StepLR scheduler
            step_size = self.config.get("step_lr_step_size", 10)
            gamma = self.config.get("step_lr_gamma", 0.5)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif self.lr_scheduler_name == "plateau":
            # Plateau scheduler
            patience = self.config.get("plateau_patience", 10)
            factor = self.config.get("plateau_factor", 0.5)
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=factor, verbose=True)
        else:
            print(f"Unknown LR scheduler: {self.lr_scheduler_name}. Using Cosine Annealing.")
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        return scheduler


    def train(self):
        """ Method to start the training process of the model."""
        # Name the model file with a timestamp depending on architecture
        model_name = f"sbr_{self.architecture}_{timestamp}.pt"
        # Move the model and noise model to the configured device (GPU or CPU)
        self.model.to(self.device)
        self.noise_model.to(self.device)
        self._set_random_seed()  # Set the random seed

        # Initialize the optimizer and learning rate scheduler
        optimizer = self._initialize_optimizer()
        scheduler = self._initialize_lr_scheduler(optimizer)
        start_time = time.time()  # Record the start time for training

        # Check if using Automatic Mixed Precision (AMP) for training
        if self.use_amp:
            print("Using mixed-precision training with AMP.")

        # Training loop for the specified number of epochs
        for epoch in range(self.epochs):
            self.model.train()  # Set the model to training mode
            total_loss = 0
            # Iterate over the training dataset
            for lf_view_stack, rfv, gt in self.train_data_loader:
                # Move the data to the configured device
                lf_view_stack, rfv, gt = (
                    lf_view_stack.to(self.device),
                    rfv.to(self.device),
                    gt.to(self.device),
                )

                # Apply the noise model to the data
                lf_view_stack, rfv = self.noise_model(lf_view_stack, rfv)

                optimizer.zero_grad()  # Reset gradients

                # Training step with or without AMP
                if self.use_amp:
                    with autocast():
                        output = self.model(lf_view_stack, rfv)
                        loss = self.criterion(output, gt)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    output = self.model(lf_view_stack, rfv)
                    loss = self.criterion(output, gt)
                    loss.backward()
                    optimizer.step()
                
                # Log the loss for the current batch
                logger.debug(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {loss.item()}")
                total_loss += loss.item()

            # Calculate and log the average training loss for the epoch
            avg_train_loss = total_loss / len(self.train_data_loader)
            self.training_losses.append(avg_train_loss)
            logger.info(f"Epoch [{epoch + 1}/{self.epochs}], Train Loss: {avg_train_loss}")

            # Perform validation and log the validation loss
            val_loss = self.validate()
            self.validation_losses.append(val_loss)
            logger.info(f"Epoch [{epoch + 1}/{self.epochs}], Validation Loss: {val_loss}")

            # Update the learning rate scheduler
            if self.lr_scheduler_name == "plateau": 
                scheduler.step(val_loss)  # For Plateau scheduler, pass validation loss as an argument
            else:
                scheduler.step()

            # Save the model if the validation loss is the lowest so far
            if val_loss < self.lowest_val_loss:
                self.lowest_val_loss = val_loss
                save_state = {
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "training_losses": self.training_losses,
                    "validation_losses": self.validation_losses,
                    "time_elapsed": time.time() - start_time,
                }

                save_state.update(self.config)
                model_save_path = os.path.join(self.model_dir, model_name)
                torch.save(save_state, model_save_path)
                logger.info("Model saved at epoch {}".format(epoch + 1))


    def validate(self):
        """
        Method to validate the model on the validation dataset.

        Returns:
            The average loss on the validation dataset.
        """
        # Set the model to evaluation mode.
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            # Iterate over the validation dataset
            for lf_view_stack, rfv, gt in self.val_data_loader:
                # Move the data to the configured device
                lf_view_stack, rfv, gt = (
                    lf_view_stack.to(self.device),
                    rfv.to(self.device),
                    gt.to(self.device),
                )

                # Forward pass through the model and calculate loss
                output = self.model(lf_view_stack, rfv)
                loss = self.criterion(output, gt)
                total_loss += loss.item()
        
        # Return the average loss across the validation dataset
        return total_loss / len(self.val_data_loader)
