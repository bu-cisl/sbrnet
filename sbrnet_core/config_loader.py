import yaml


def load_config(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Check the "backbone" parameter. unet is deprecated and removed anyways.
    if "backbone" not in config or config["backbone"] not in ["resnet", "unet"]:
        raise ValueError(
            "Invalid or missing 'backbone' parameter in the configuration file. It should be 'resnet' or 'unet'."
        )

    return config
