from omegaconf import DictConfig
import torch
import pandas as pd
import logging
from pathlib import Path
import hydra
from hydra.core.hydra_config import HydraConfig

from .model import ModelInitial, ModelInitialDropout
from .workout_dataset import load_data, load_test_data

logger: logging.Logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    model_path = Path(cfg["model"])
    test_csv = Path(cfg["data"]["test"])
    output_csv = Path(cfg["data"]["save"])

    model = ModelInitialDropout()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    ids, x_test = load_test_data(test_csv)

    with torch.no_grad():
        preds = model(x_test).squeeze().numpy()

    preds = preds.clip(min=0)

    submission = pd.DataFrame({
        "id": ids,
        "Calories": preds
    })

    submission.to_csv(output_csv, index=False)
    logger.info("Submission saved")


if __name__ == "__main__":
    main()
