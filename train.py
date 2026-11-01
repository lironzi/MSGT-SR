import yaml
import argparse
from misc.trainer import Trainer



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a MSGNet.")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml file")
    args = parser.parse_args()

    with open(args.config,'r') as f:
        config=yaml.safe_load(f)

    trainer=Trainer(config)
    trainer.train()