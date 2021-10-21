import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os
import yaml
import argparse
import glob
# import cupy as np

"""" Import helper functions """
from scf_fp_helpers import *
from experiment_helpers import *

class ExperimentRunner():

    def __init__(self):
      self.opt = None
      self.seed = None
      # self.total_comp_field = np.zeros((256, 256, 36))
      self.total_comp_field = np.zeros((256, 256, 36))

    def set_opt(self, opt):
      if opt.seed is None:
          raise ValueError(
              "The seed was not specified correctly in the config file")

      # Sets random seed from system time if seed is None
      np.random.seed(opt.seed)
      self.opt = opt
      self.seed = opt.seed
      
    def _plot_fields(self, field, field_file, field_img_file, log=False):
        plt.figure(figsize = (20, 12))
        collapsed_field = np.sum(field, axis = 2)
        if log:
          collapsed_field = np.log(collapsed_field)
        create_path(field_file)
        plt.imshow(collapsed_field)
        plt.colorbar()
        filename = f'{field_img_file}_log' if log else f'{field_img_file}_linear'
        plt.savefig(f"{self.experiments_path}/{filename}")
        plt.clf()

    def completion(self, experiments_path):
        self.experiments_path = experiments_path

        sources, sinks = np.array(self.opt.sources), np.array(self.opt.sinks)
        sources, sinks = translate_pointset(sources), translate_pointset(sinks)

        ############# Getting source fields ####################
        print("Generating source fields...")
        source_field = compute_fokker_planck(sources, w = self.opt.width, tau = self.opt.tau, sigma_sqrt = self.opt.diff_sqrt)
        if(self.opt.save_source_data):
          source_field_saved = np.array(source_field)
          np.savez_compressed(f"{self.experiments_path}/{self.opt.source_field_data}", data=source_field_saved)
        if(self.opt.plot_source_fields):
          source_field_plot = np.array(source_field)
          self._plot_fields(source_field_plot, self.opt.source_file, self.opt.source_img_file)
          self._plot_fields(source_field_plot, self.opt.source_file, self.opt.source_img_file, log=True)

        ############# Getting sink fields ####################
        print("Generating sink fields...")
        sink_field = compute_fokker_planck(sinks, w = self.opt.width, tau = self.opt.tau, sigma_sqrt = self.opt.diff_sqrt, src=False)
        if(self.opt.save_sink_data):
          sink_field_saved = np.array(sink_field)
          np.savez_compressed(f"{self.experiments_path}/{self.opt.sink_field_data}", data=sink_field_saved)
        if(self.opt.plot_sink_fields):
          sink_field_plot = np.array(sink_field)
          self._plot_fields(sink_field_plot, self.opt.sink_file, self.opt.sink_img_file)
          self._plot_fields(sink_field_plot, self.opt.sink_file, self.opt.sink_img_file, log=True)

        ############ Getting completion fields ####################
        print("Generating completion fields...")
        comp_field = np.multiply(source_field, sink_field)
        if(self.opt.save_completion_data):
          comp_field_saved = np.array(comp_field)
          np.savez_compressed(f"{self.experiments_path}/{self.opt.completion_field_data}", data=comp_field_saved)
        if(self.opt.plot_completion_fields):
          comp_field_plot = np.array(comp_field)
          self._plot_fields(comp_field_plot, self.opt.completion_file, self.opt.completion_img_file)
          self._plot_fields(comp_field_plot, self.opt.completion_file, self.opt.completion_img_file, log=True)

        self.total_comp_field += comp_field


def main(args):
    runner = ExperimentRunner()
    create_path(args.experiment_dir)
    full_config_path = f"{args.config_file}"
    print(f"Running {full_config_path}")
    opt = get_config_no_saving(full_config_path)
    runner.set_opt(opt)
    runner.completion(args.experiment_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--experiment_dir', type=str, required=True)

    args = parser.parse_args()

    print('Arguments:\n{}\n'.format(' '.join(sys.argv[1:])))

    main(args)
