import os
import sys
import yaml
from types import SimpleNamespace

def create_path(path, directory=False):
    if directory:
        dir = path
    else:
        dir = os.path.dirname(path)
    if not os.path.exists(dir):
        print(f'{dir} does not exist, creating')
        try:
            os.makedirs(dir)
        except Exception as e:
            print(e)
            print(f'Could not create path {path}')

def get_config(experiment_dir, config_file):
    base_path = experiment_dir
    create_path(base_path, directory=True)

    with open(config_file) as f:
        if not os.path.isfile(config_file):
            print(f"File {config_file} does not exist")
            sys.exit(0)
        data = yaml.load(f, Loader=yaml.FullLoader)

    result = SimpleNamespace(**data)

    # save a copy of config for documentation purposes
    with open(f"{base_path}/config.yml", "w") as f:
        f.write(yaml.dump(data))

    result.walks_file = f"{base_path}/{result.walks_file}"
    result.walks_img_file = f"{base_path}/{result.walks_img_file}"
    result.sink_field_file = f"{base_path}/{result.sink_field_file}"
    result.sink_img_file = f"{base_path}/{result.sink_img_file}"
    result.source_field_file = f"{base_path}/{result.source_field_file}"
    result.source_img_file = f"{base_path}/{result.source_img_file}"
    result.completion_field_file = f"{base_path}/{result.completion_field_file}"
    result.completion_img_file = f"{base_path}/{result.completion_img_file}"

    result.base_path = base_path

    return result