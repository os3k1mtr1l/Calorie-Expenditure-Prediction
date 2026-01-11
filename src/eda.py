from omegaconf import DictConfig
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import hydra
from hydra.core.hydra_config import HydraConfig

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    df = pd.read_csv(cfg["data"]["train"])
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0}).astype(float)
    save_path = Path(HydraConfig.get().run.dir)

    plt.scatter(df["Age"], df["Calories"], alpha=0.3)
    plt.xlabel("Age")
    plt.ylabel("Calories")
    plt.title("Age vs Calories")
    plt.savefig(str(save_path / "age_calories.png"))

    plt.scatter(df["Weight"], df["Calories"], alpha=0.3)
    plt.xlabel("Weight")
    plt.ylabel("Calories")
    plt.title("Weight vs Calories")
    plt.savefig(str(save_path / "weight_calories.png"))

    corr = df.drop(columns=["id"]).corr()
    print(corr["Calories"].sort_values(ascending=False))

    plt.imshow(corr, cmap="coolwarm")
    plt.colorbar()
    plt.xticks(range(len(corr)), corr.columns, rotation=90)
    plt.yticks(range(len(corr)), corr.columns)
    plt.title("Correlation matrix")
    plt.savefig(str(save_path / "cor_mat.png"))

if __name__ == "__main__":
    main()