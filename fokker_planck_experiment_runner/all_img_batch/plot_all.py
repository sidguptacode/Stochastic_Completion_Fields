from scipy.io import loadmat
import numpy as np
from skimage.draw import line
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import argparse
from tqdm import tqdm, trange
import numpy.ma as ma


def render_lines(strokes, shape=(256, 256)):
    img = np.ones(shape, dtype=np.uint8) * 255
    for stroke in strokes:
        lines = stroke
        for x1, y1, x2, y2, orientation, length in lines:
            rr, cc = line(int(y1) // 8, int(x1) // 8,
                          int(y2) // 8, int(x2) // 8)
            img[rr, cc] = 0
    return img // 255


def _plot_img(img, imgpath, log=False):
    plt.figure(figsize=(20, 12))
    plt.imshow(img, cmap='gray')
    plt.colorbar()
    plt.savefig(imgpath)
    plt.clf()


def _plot_fields(field, imgpath, log=False):
    plt.figure(figsize=(20, 12))
    if log:
        field = np.log(field)
    if len(field.shape) == 3:
        field = field.sum(axis=2)
    plt.imshow(field)
    plt.colorbar()
    plt.savefig(imgpath)
    plt.clf()


def _plot_overlay(img1, img2, imgpath, log=False):
    plt.figure(figsize=(20, 12))
    # plt.imshow(img1, cmap='gray')
    plt.imshow(img2, cmap='gray')
    plt.colorbar()
    plt.savefig(imgpath)
    plt.clf()


def _plot_subplot(middle_img, junction_img, mid_scf, junc_scf, mid_overlay, junc_overlay, imgpath, log=False):
    plt.figure(figsize=(12, 20))
    plt.subplot(3, 2, 1)
    plt.imshow(middle_img, cmap='gray')
    plt.title("Middle image", fontsize=18)
    plt.colorbar()
    plt.subplot(3, 2, 2)
    plt.imshow(junction_img, cmap='gray')
    plt.title("Junction image", fontsize=18)
    plt.colorbar()
    plt.subplot(3, 2, 3)
    plt.imshow(mid_scf, cmap='gray')
    plt.title("Middle SCF", fontsize=18)
    plt.colorbar()
    plt.subplot(3, 2, 4)
    plt.imshow(junc_scf, cmap='gray')
    plt.title("Junction SCF", fontsize=18)
    plt.colorbar()
    plt.subplot(3, 2, 5)
    plt.imshow(mid_overlay, cmap='gray')
    plt.title("Middle image-SCF overlay", fontsize=18)
    plt.colorbar()
    plt.subplot(3, 2, 6)
    plt.imshow(junc_overlay, cmap='gray')
    plt.title("Junction image-SCF overlay", fontsize=18)
    plt.colorbar()
    plt.savefig(imgpath)
    plt.clf()


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--sandvld_directory', type=str, required=True)
    parser.add_argument('--experiments_dir', type=str, required=True)
    args = parser.parse_args()

    # line_drawings_middle = loadmat(
    #     f'{args.sandvld_directory}/SandVLDs_middle_equalLength.mat')
    # line_drawings_junctions = loadmat(
    #     f'{args.sandvld_directory}/SandVLDs_junction_equalLength.mat')
    # line_drawings_full = loadmat(
    #     f'{args.sandvld_directory}/SandVLDs_cleaned.mat')
    # LD_middle = line_drawings_middle["middleLD"]
    # LD_junction = line_drawings_junctions["junctionLD"]
    # LD_full = line_drawings_full["LD"]

    for i in trange(0, 260):
        # for i in range(19, 20):
        # Plotting base images
        # full_figure = LD_full[0][i]
        # full_strokes = full_figure[2][0]
        # full_img = render_lines(full_strokes)
        # full_filepath = f"{args.experiments_dir}/ld_{i}/full_img"

        # middle_figure = LD_middle[0][i]
        # middle_strokes = middle_figure[2][0]
        # middle_img = render_lines(middle_strokes)
        # middle_img = middle_img
        # middle_filepath = f"{args.experiments_dir}/ld_{i}/middle_img"

        # junction_figure = LD_junction[0][i]
        # junction_strokes = junction_figure[2][0]
        # junction_img = render_lines(junction_strokes)
        # junction_img = junction_img
        # junction_filepath = f"{args.experiments_dir}/ld_{i}/junction_img"

        # _plot_img(full_img, f"{full_filepath}.png")
        # _plot_img(middle_img, f"{middle_filepath}.png")
        # _plot_img(junction_img, f"{junction_filepath}.png")

        # Plotting SCF's
        base_path = f"{args.experiments_dir}/ld_{i}"
        mid_scf_path = f"{base_path}/middle"
        mid_scf = np.load(f"{mid_scf_path}.npz", allow_pickle=True)["data"]
        mid_scf = np.array(mid_scf).sum(axis=2)
        # mid_scf /= np.max(mid_scf)
        _plot_fields(mid_scf, f"{mid_scf_path}_scf.png")
        _plot_fields(mid_scf, f"{mid_scf_path}_log_scf.png", log=True)

        junc_scf_path = f"{base_path}/junc"
        junc_scf = np.load(f"{junc_scf_path}.npz", allow_pickle=True)["data"]
        junc_scf = np.array(junc_scf).sum(axis=2)
        # junc_scf /= np.max(junc_scf)
        _plot_fields(junc_scf, f"{junc_scf_path}_scf.png")
        _plot_fields(junc_scf, f"{junc_scf_path}_log_scf.png", log=True)

        # Plotting overlays
        # TODO: Try using np.ma
        # mask = ma.masked_where((1 - junc_scf)>0, (1 - junc_scf))
        # junc_scf_mask = ma.masked_array(1 - junc_scf,mask)
        # mask = ma.masked_where(mid_scf==1.0, mid_scf)
        # mid_scf_mask = ma.masked_array(mid_scf,mask)
        # _plot_overlay(junction_img, junc_scf_mask, f"{junc_scf_path}_overlay.png")
        # _plot_overlay(middle_img, mid_scf_mask, f"{mid_scf_path}_overlay.png")

        # mid_overlay = middle_img + (1 - mid_scf)
        # junc_overlay = junction_img + (1 - junc_scf)
        # mid_overlay /= np.max(mid_overlay)
        # junc_overlay /= np.max(junc_overlay)

        # _plot_img(mid_overlay, f"{junc_scf_path}_overlay.png")
        # _plot_img(junc_overlay, f"{mid_scf_path}_overlay.png")

        # Black background, white lines
        # _plot_subplot(middle_img, junction_img, mid_scf, junc_scf, mid_overlay, junc_overlay, f"{base_path}/all_plots.png")
        # White background, black lines
        # _plot_subplot(middle_img, junction_img, 1 - mid_scf, 1 - junc_scf,
        #               mid_overlay, junc_overlay, f"{base_path}/all_plots.png")


if __name__ == '__main__':
    main()
