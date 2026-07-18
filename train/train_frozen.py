import hydra

from train.common import run_training


@hydra.main(version_base=None, config_path="../configs", config_name="frozen")
def main(cfg):
    run_training(cfg, strategy="frozen")


if __name__ == "__main__":
    main()
