import yaml
import os

def load_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config


def load_all_configs(config_dir="configs"):

    configs = {}

    for file in os.listdir(config_dir):
        if file.endswith(".yaml") or file.endswith(".yml"):
            path = os.path.join(config_dir, file)
            name = file.split(".")[0]
            configs[name] = load_config(path)

    return configs