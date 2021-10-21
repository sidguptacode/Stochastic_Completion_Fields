import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os
import json
import yaml
import argparse

"""" Import helper functions """
from scf_helpers import random_walks, discretize_walks, rotate_and_translate, matrix_point
from experiment_helpers import create_path, get_config

""" Import constants """
from scf_helpers import RADIANS_PER_ROTATION, DEGREES_PER_RADIAN

class ExperimentRunner():
    def __init__(self, opt):

      if opt.seed is None:
          raise ValueError(
              "The seed was not specified correctly in the config file")

      # Sets random seed from system time if seed is None
      np.random.seed(opt.seed)

      self.opt = opt
      self.seed = opt.seed

    def walks(self):
      print(f'Generating {self.opt.num_walks} random walks')
      walks = random_walks(self.opt.num_walks)

      if(self.opt.plot_walks):
        dwalks = discretize_walks(walks)
        plt.imshow(np.log(np.sum(dwalks, axis=2)), interpolation='none')
        create_path(self.opt.walks_file)
        plt.savefig(f'{self.opt.walks_img_file}')
        plt.clf()

      print(f'Saving {self.opt.num_walks} random walks trace to: {self.opt.walks_file}.json')
      create_path(self.opt.walks_file)

      walks_data = {}
      walks_data["data"] = walks

      with open(f'{self.opt.walks_file}.json', 'w') as fp:
        json.dump(walks_data, fp)

    def completion(self):
      print(f'Loading walks from {self.opt.walks_file}.json')
      with open(f'{self.opt.walks_file}.json') as json_file:
        walks_data = json.load(json_file)
        walks = walks_data["data"]

      # only considering one sink and one source at the moment

      # Source Field (Load or Compute)
      if (self.opt.load_source_field_from_file):
        with open(f'{self.opt.source_field_file}.json') as json_file:
            source_data = json.load(json_file)
            source_field = source_data["data"]
      else: 
        u = (self.opt.sources[0]["x"], self.opt.sources[0]["y"], self.opt.sources[0]["theta"])
        print("Source: ", u)
        pU = np.zeros((256, 256, 36))
        pU[matrix_point(u)] = 1
        x, y, theta = u
        theta = -theta / DEGREES_PER_RADIAN
        source_rotated_walks = rotate_and_translate(walks, x, y, theta, origin=(0, 0))
        source_field = discretize_walks(source_rotated_walks)

        print(f'Saving source field to: {self.opt.source_field_file}.json')
        create_path(self.opt.source_field_file)
        with open(f'{self.opt.source_field_file}.json', 'w') as json_file:
          source_data = {}
          source_data["data"] = source_field.tolist()
          json.dump(source_data, json_file)

      # Sink Field (Load or Compute)
      if (self.opt.load_sink_field_from_file):
        with open(f'{self.opt.sink_field_file}.json') as json_file:
            sink_data = json.load(json_file)
            sink_field = sink_data["data"]
      else: 
        v = (self.opt.sinks[0]["x"], self.opt.sinks[0]["y"], self.opt.sinks[0]["theta"])
        print("Sink: ", v)
        pV = np.zeros((256, 256, 36))
        pV[matrix_point(v)] = 1
        x, y, theta = v
        theta = -theta / DEGREES_PER_RADIAN
        sink_rotated_walks = rotate_and_translate(walks, x, y, theta, origin=(0, 0))
        sink_field = discretize_walks(sink_rotated_walks)

        print(f'Saving sink field to: {self.opt.sink_field_file}.json')
        create_path(self.opt.sink_field_file)
        with open(f'{self.opt.sink_field_file}.json', 'w') as json_file:
            sink_data = {}
            sink_data["data"] = sink_field.tolist()
            json.dump(sink_data, json_file)
      
      # Completion field (Compute)
      comp_field = np.multiply(source_field, sink_field)
      
      print(f'Saving completion field to: {self.opt.completion_field_file}.json')
      create_path(self.opt.completion_field_file)
      with open(f'{self.opt.completion_field_file}.json', 'w') as json_file:
          comp_data = {}
          comp_data["data"] = comp_field.tolist()
          json.dump(comp_data, json_file)

      if(self.opt.plot_fields):
        print(f'Saving Source Field Image to: {self.opt.source_img_file}.png')
        create_path(self.opt.source_img_file)
        plt.imshow(np.log(np.sum(source_field, axis=2)), interpolation='none')
        plt.savefig(f'{self.opt.source_img_file}')
        plt.clf()
        plt.close()
        
        print(f'Saving Sink Field Image to: {self.opt.sink_img_file}.png')
        create_path(self.opt.sink_img_file)
        plt.imshow(np.log(np.sum(sink_field, axis=2)), interpolation='none')
        plt.savefig(f'{self.opt.sink_img_file}')
        plt.clf()

        print(f'Saving Completion Field Image to: {self.opt.completion_img_file}.png')
        create_path(self.opt.completion_img_file)
        plt.imshow(np.log(np.sum(comp_field, axis=2)), interpolation='none')
        plt.savefig(f'{self.opt.completion_img_file}')
        plt.clf()
    
def main(args):
    opt = get_config(args.experiment_dir, args.config_file)
    runner = ExperimentRunner(opt)

    if args.mode == "walks":
        runner.walks()
    elif args.mode == "completion":
        runner.completion()
    elif args.mode == "both":
        runner.walks()
        runner.completion()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Probprog")
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--experiment_dir', type=str, required=True)
    parser.add_argument('--mode', default=None, type=str, choices=[
        "walks",
        "completion",
        "both"
    ], required=True)

    args = parser.parse_args()

    print('Arguments:\n{}\n'.format(' '.join(sys.argv[1:])))

    main(args)