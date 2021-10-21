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

def get_config_no_saving(config_file):
    with open(config_file) as f:
        if not os.path.isfile(config_file):
            print(f"File {config_file} does not exist")
            sys.exit(0)
        data = yaml.load(f, Loader=yaml.FullLoader)
    result = SimpleNamespace(**data)
    return result

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

    result.sink_file = f"{base_path}/{result.sink_file}"
    result.sink_img_file = f"{base_path}/{result.sink_img_file}"
    result.sink_field_data = f"{base_path}/{result.sink_field_data}"
    result.source_file = f"{base_path}/{result.source_file}"
    result.source_img_file = f"{base_path}/{result.source_img_file}"
    result.source_field_data = f"{base_path}/{result.source_field_data}"
    result.completion_file = f"{base_path}/{result.completion_file}"
    result.completion_img_file = f"{base_path}/{result.completion_img_file}"
    result.completion_field_data = f"{base_path}/{result.completion_field_data}"
    result.source_sink_sum_file = f"{base_path}/{result.source_sink_sum_file}"
    result.source_sink_sum_img_file = f"{base_path}/{result.source_sink_sum_img_file}"
    result.base_path = base_path

    return result


