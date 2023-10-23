import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
import time
from torch.cuda.amp import GradScaler, autocast

from sbrnet_core.sbrnet.dataset import CustomDataset, MySubset


class Trainer:
    def __init__(
        self,
        model: Module,
        config,
    ):
        self.config = config
        self.model = model
        self.learning_rate = config["learning_rate"]
        self.epochs = config["epochs"]
        self.model_save_path = config["model_save_path"]
        self.lowest_val_loss = float("inf")
        self.training_losses = []
        self.validation_losses = []
        self.random_seed = config.get("random_seed", None)
        self.use_amp = config.get("use_amp", False)
        self.optimizer_name = config.get("optimizer", "adam")
        self.lr_scheduler_name = config.get("lr_scheduler", "cosine_annealing")
        self.criterion_name = config.get("loss_criterion", "bce_with_logits")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.scaler = (
            GradScaler() if self.use_amp else None
        )  # Initialize the scaler if using AMP

        # Initialize the loss criterion based on the configuration
        if self.criterion_name == "bce_with_logits":
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.criterion_name == "mse":
            self.criterion = nn.MSELoss()
        elif self.criterion_name == "mae":
            self.criterion = nn.L1Loss()
        else:
            print(
                f"Unknown loss criterion: {self.criterion_name}. Using BCEWithLogitsLoss."
            )

        complete_dataset: Dataset = CustomDataset(config["dataset_path"])

    def set_random_seed(self):
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_seed)

    def initialize_optimizer(self):
        if self.optimizer_name == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == "sgd":
            optimizer = optim.SGD(
                self.model.parameters(), lr=self.learning_rate, momentum=0.9
            )
        else:
            print(f"Unknown optimizer: {self.optimizer_name}. Using Adam.")
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def initialize_lr_scheduler(self, optimizer):
        if self.lr_scheduler_name == "cosine_annealing":
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
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
        self.set_random_seed()

        optimizer = self.initialize_optimizer()
        scheduler = self.initialize_lr_scheduler(optimizer)
        start_time = time.time

        if self.use_amp:
            print("Using mixed-precision training with AMP.")

        for epoch in range(self.epochs):
            self.model.to(self.device)
            self.model.train()
            total_loss = 0
            for lf_view_stack, rfv, gt in self.train_data_loader:
                lf_view_stack, rfv, gt = (
                    lf_view_stack.to(self.device),
                    rfv.to(self.device),
                    gt.to(self.device),
                )

                optimizer.zero_grad()

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

                total_loss += loss.item()

            avg_train_loss = total_loss / len(self.train_data_loader)
            self.training_losses.append(avg_train_loss)
            print(f"Epoch [{epoch + 1}/{self.epochs}], Train Loss: {avg_train_loss}")

            val_loss = self.validate()
            self.validation_losses.append(val_loss)
            print(f"Epoch [{epoch + 1}/{self.epochs}], Validation Loss: {val_loss}")

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
                    "time_elapsed": time.time() - start_time,
                }
                torch.save(save_state, self.model_save_path)

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for lf_view_stack, rfv, gt in self.val_data_loader:
                lf_view_stack, rfv, gt = (
                    lf_view_stack.to(self.device),
                    rfv.to(self.device),
                    gt.to(self.device),
                )
                output = self.model(lf_view_stack, rfv)
                loss = self.criterion(output, gt)
                total_loss += loss.item()
        return total_loss / len(self.val_data_loader)
