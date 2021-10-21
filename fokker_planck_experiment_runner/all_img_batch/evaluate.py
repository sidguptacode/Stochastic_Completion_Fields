import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat
from skimage.draw import line
from tqdm import trange, tqdm
import argparse

def render_lines(strokes, divisor=1, shape=(1500, 2048)):
  img = np.zeros(shape, dtype=np.uint8) * 255
  for stroke in strokes:
    lines = stroke
    for x1, y1, x2, y2, orientation, length in lines: 
      rr, cc = line(int(x1 // divisor), int(y1 // divisor), int(x2 // divisor), int(y2 // divisor))
      img[cc,rr] = 1
  return img


def main():

  parser = argparse.ArgumentParser(description="")
  parser.add_argument('--sandvld_directory', type=str, required=True)
  parser.add_argument('--experiments_dir', type=str, required=True)
  args = parser.parse_args()

  line_drawings_middle = loadmat(f'{args.sandvld_directory}/SandVLDs_middle_equalLength.mat')
  line_drawings_junctions = loadmat(f'{args.sandvld_directory}/SandVLDs_junction_equalLength.mat')
  line_drawings_full = loadmat(f'{args.sandvld_directory}/SandVLDs_cleaned.mat')
  LD_middle = line_drawings_middle["middleLD"]
  LD_junction = line_drawings_junctions["junctionLD"]
  LD_full = line_drawings_full["LD"]

  num_mid, num_junc = 0, 0

  for i in range(260):
    figure_middle = LD_middle[0][i]
    strokes_middle = figure_middle[2][0]
    figure_junction = LD_junction[0][i]
    strokes_junc = figure_junction[2][0]

    middle_img = render_lines(strokes_middle, divisor=8, shape=(256, 256))
    junc_img = render_lines(strokes_junc, divisor=8, shape=(256, 256))

    base_path = f"{args.experiments_dir}/ld_{i}"
    mid_scf_path = f"{base_path}/middle"
    junc_scf_path = f"{base_path}/junc"

    mid_completion = np.load(f"{mid_scf_path}.npz",allow_pickle=True)['data'].sum(axis=2)
    junc_completion = np.load(f"{junc_scf_path}.npz",allow_pickle=True)['data'].sum(axis=2)

    # Perhaps we should normalize by the number of pixels that are in each condition
    # I.e, number of non-zero pixels in the target
    comp_for_mid = np.multiply(junc_img, mid_completion)
    comp_for_junc = np.multiply(middle_img, junc_completion)

    mid_prob, junc_prob = np.sum(comp_for_mid), np.sum(comp_for_junc) 

    max_prob = max(mid_prob, junc_prob)

    mid_prob_normal = mid_prob / max_prob
    junc_prob_normal = junc_prob / max_prob

    diff_prob = mid_prob_normal - junc_prob_normal

    # print(f'Image={i}\t|\t{mid_prob}\t|\t{junc_prob}\t|\t')
    print(f'Image={i}\t|\t{diff_prob}')
    if mid_prob > junc_prob:
        num_mid+=1
    else:
        num_junc+=1

  print(f'Middle counts={num_mid}\t|\tJunction counts={num_junc}')

main()


