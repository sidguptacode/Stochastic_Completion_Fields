import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import argparse
import os

# Importing code from the parent directory
import sys
sys.path.append("..")
from fokker_planck_experiment_runner.experiment_helpers import create_path

def _plot_fields(field, imgpath, log=False):
    plt.figure(figsize = (20, 12))
    if log:
        field = np.log(field)
    plt.imshow(field)
    plt.colorbar()
    plt.savefig(imgpath)
    plt.clf()


def main(img_experiments_dir, contour_type, save_field=True):

    comp_field = np.zeros((256, 256, 36))

    all_subdirs = []
    for subdir, _, _ in os.walk(img_experiments_dir):
        all_subdirs.append(subdir)
    all_subdirs.pop(0)

    # Total number of distributions
    num_dists = len(all_subdirs)

    for i in trange(num_dists):
        curr_dir = all_subdirs[i]
        comp_data = np.load(f"{curr_dir}/completion_field_data.npz",allow_pickle=True)
        comp_loaded = np.array(comp_data["data"])
        comp_field += comp_loaded

    comp_field = np.sum(comp_field, axis = 2)
    comp_field *= (1 / num_dists)


    marginalized_dir = f"{img_experiments_dir}/../marginalized"
    create_path(marginalized_dir, directory=True)

    if contour_type == "middle":
        results_path = f"{marginalized_dir}/middle"
    else:
        results_path = f"{marginalized_dir}/junc"

    _plot_fields(comp_field, f"{results_path}.png")

    if (save_field):
        np.savez_compressed(results_path, data=comp_field)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--img_experiments_dir', type=str, required=True)
    parser.add_argument('--contour_type', type=str, required=True)
    args = parser.parse_args()

    main(args.img_experiments_dir, args.contour_type, save_field=True)
