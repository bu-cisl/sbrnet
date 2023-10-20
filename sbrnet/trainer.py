import torch
import torch.optim as optim
import torch.nn as nn
import yaml


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


class Trainer:
    def __init__(self, model, data_loader, config):
        self.model = model
        self.data_loader = data_loader
        self.learning_rate = config["learning_rate"]
        self.epochs = config["epochs"]
        self.model_save_path = config["model_save_path"]
        self.lowest_val_loss = float("inf")

    def train(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for batch_idx, (data, target) in enumerate(self.data_loader):
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(
                f"Epoch [{epoch + 1}/{self.epochs}], Train Loss: {total_loss / len(self.data_loader)}"
            )

            # Validation loop
            val_loss = self.validate()
            print(f"Epoch [{epoch + 1}/{self.epochs}], Validation Loss: {val_loss}")

            # Save the model if validation loss is the lowest so far
            if val_loss < self.lowest_val_loss:
                self.lowest_val_loss = val_loss
                torch.save(self.model.state_dict(), self.model_save_path)

    def validate(self):
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.data_loader):
                output = self.model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
        return total_loss / len(self.data_loader)
