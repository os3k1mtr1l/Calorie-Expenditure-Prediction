import logging
from pathlib import Path

import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
from matplotlib import pyplot as plt
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from .workout_dataset import WorkoutDataset
from .model import ModelInitial, ModelSmall, ModelLarge, ModelInitialDropout
from .rmsle_loss import RMSLELoss


logger: logging.Logger = logging.getLogger(__name__)

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module
) -> float:
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X, y in dataloader:
            preds = model(X)
            loss = criterion(preds, y)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    model_type: str,
    epochs: int = 100,
    lr: float = 1e-3,
    momentum: float = 0.9,
    seed: int = 72
) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    save_path = Path(HydraConfig.get().run.dir)

    criterion = RMSLELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    best_rmsle = float("inf")
    train_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            preds = model(X_batch)
            
            loss = criterion(preds, y_batch)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

        running_loss /= len(train_loader)
        train_losses.append(running_loss)

        # if epoch % 10 == 0:
        val_rmsle = evaluate(model, val_loader, criterion)

        if val_rmsle < best_rmsle:
            best_rmsle = val_rmsle
            torch.save(model.state_dict(), str(save_path / "best_model.pth"))

        logger.info(
            f"Epoch [{epoch+1}/{epochs}] "
            f"TRAIN LOSS: {running_loss:.4f} "
            f"VAL RMSLE: {val_rmsle:.4f}"
        )

    plt.plot(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt_save = str(save_path / f"{model_type}_graph.png")
    plt.savefig(plt_save)
    # plt.show()

def prep_phase(cfg: DictConfig, stage: str = '') -> None:
    # Przygotowanie DataLoader'a z danymi
    dataset: Dataset = WorkoutDataset(cfg["data"]["train"])

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg["training"]["seed"])
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False
    )

    # Zdefiniowanie modeli
    # models: list[nn.Module] = [ModelInitial, ModelSmall, ModelLarge]
    # models: list[nn.Module] = [ModelInitial, ModelInitialDropout]
    models: list[nn.Module] = [ModelInitial]
    
    # Porównanie modeli
    for model in models:
        logger.info(f"{model.__name__}")
        # Trenowanie modelu
        train(
            train_loader=train_loader,
            val_loader=val_loader,
            model=model(),
            model_type=(model.__name__ + stage),
            epochs=cfg["training"]["epochs"],
            lr=cfg["training"]["learning_rate"],
            momentum=cfg["training"]["momentum"],
            seed=cfg["training"]["seed"]
        )

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg : DictConfig) -> None:
    """ # Porównanie hyperparametrów
    hyperparam: dict[str, list] = {
        "learning_rate": [1e-2, 1e-4, 1e-3],
        "batch_size": [32, 64, 16],
        "momentum": [0.0, 0.9]
    }

    for key, values in hyperparam.items():
        for value in values:
            logger.info(
                f"Hyperparameter: {key} "
                f"Value: {value}"
            )

            cfg["training"][key] = value
            stage = f" {key}_{value}"

            prep_phase(cfg, stage)

    """
    prep_phase(cfg)

if __name__ == "__main__":
    main()