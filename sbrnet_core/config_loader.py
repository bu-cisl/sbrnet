import yaml


def load_config(config_path: str):
    """
    Load a configuration file in YAML format and validate its contents.

    Args:
        config_path (str): The file path of the YAML configuration file.

    Returns:
        A dictionary containing the configuration settings.
    """

    # Open and read the YAML configuration file
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Check the "backbone" parameter. unet is deprecated and removed anyways.
    if "backbone" not in config or config["backbone"] not in ["resnet", "unet", "densenet", "efficientnet"]:
        raise ValueError(
            "Invalid or missing 'backbone' parameter in the configuration file. It should be 'resnet' or 'unet'."
        )

    return config
