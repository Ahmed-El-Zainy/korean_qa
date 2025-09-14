import yaml


def load_yaml_config(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config



if __name__=="__main__":
    print(f"config ...")