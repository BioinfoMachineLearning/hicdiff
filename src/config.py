import argparse
import os
import yaml

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

if __name__ == '__main__':
    # parse config file
    with open(os.path.join("../configs", "bedroom.yml"), "r") as f:
        config = yaml.safe_load(f)

    print(config)
    new_config = dict2namespace(config)
    print("\n")
    print(new_config)
    print(new_config.data.dataset)

