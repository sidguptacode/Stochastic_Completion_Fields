from scipy.io import loadmat
import numpy as np
from skimage.draw import line
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import argparse
from tqdm import tqdm, trange
import numpy.ma as ma
import scipy.io


def render_lines(strokes, shape=(256, 256)):
    img = np.ones(shape, dtype=np.uint8) * 255
    for stroke in strokes:
        lines = stroke
        for x1, y1, x2, y2, orientation, length in lines:
            rr, cc = line(int(y1) // 8, int(x1) // 8,
                          int(y2) // 8, int(x2) // 8)
            img[rr, cc] = 0
    return img // 255


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--sandvld_directory', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    line_drawings_middle = loadmat(
        f'{args.sandvld_directory}/SandVLDs_middle_equalLength.mat')
    line_drawings_junctions = loadmat(
        f'{args.sandvld_directory}/SandVLDs_junction_equalLength.mat')
    line_drawings_full = loadmat(
        f'{args.sandvld_directory}/SandVLDs_cleaned.mat')
    LD_middle = line_drawings_middle["middleLD"]
    LD_junction = line_drawings_junctions["junctionLD"]
    LD_full = line_drawings_full["LD"]

    for i in trange(0, 260):
        # Plotting base images
        full_figure = LD_full[0][i]
        full_strokes = full_figure[2][0]
        full_img = render_lines(full_strokes)
        full_filepath = f"{args.output_dir}/ld_{i}/full_img"

        middle_figure = LD_middle[0][i]
        middle_strokes = middle_figure[2][0]
        middle_img = render_lines(middle_strokes)
        middle_img = middle_img
        middle_filepath = f"{args.output_dir}/ld_{i}/middle_img"

        junction_figure = LD_junction[0][i]
        junction_strokes = junction_figure[2][0]
        junction_img = render_lines(junction_strokes)
        junction_img = junction_img
        junction_filepath = f"{args.output_dir}/ld_{i}/junction_img"

        full_img = np.asarray(full_img)
        scipy.io.savemat(f"{full_filepath}.mat", {"data": full_img})

        middle_img = np.asarray(middle_img)
        scipy.io.savemat(f"{middle_filepath}.mat", {"data": middle_img})

        junction_img = np.asarray(junction_img)
        scipy.io.savemat(f"{junction_filepath}.mat", {"data": junction_img})


if __name__ == '__main__':
    main()
