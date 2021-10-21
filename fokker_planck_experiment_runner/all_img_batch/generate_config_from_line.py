from scipy.io import loadmat
import numpy as np
from skimage.draw import line
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import argparse

# Importing code from the parent directory
import sys
sys.path.append("..")
from fokker_planck_experiment_runner.experiment_helpers import create_path

def get_sources_and_sinks(strokes, junctions=False):
  # y1,x1 is going to be a sink and y2,x2 is going to be a source.
  # For every line, the source will connect to the sink from the next line (except for the
  # very last source; which will connect to the very first sink)
  sources = []
  sinks = []
  for stroke in strokes:
    lines = stroke
    first_line = lines[0]
    last_line = lines[-1]
    x1, y1, _, _, orientation1, _ = first_line
    _, _, x2, y2, orientation2, _ = last_line

    # Keep in mind that in the image space, positive y goes down.
    # We reflect angles on the horizontal axis to show this.
    orientation1 = 360.0 - orientation1
    orientation2 = 360.0 - orientation2

    if orientation2 == 360.0:
        orientation2 = 0.0

    sinks.append([int(y1) // 8, int(x1) // 8, (orientation1 + 180) % 360])
    sources.append([int(y2) // 8, int(x2) // 8, orientation2])

  ## TODO: Add back if we are generating the fields. Not for the tracing algorithm.
  # if junctions:
  #   # Just as a double-check, remove all corners detected as source/sinks
  #   # A corner is registered as a keypoint who is both a source and a sink (without consideration of the angle).
  #   all_keypts = np.array([ source[0:2] for source in sources ] + [ sink[0:2] for sink in sinks ])
  #   unique_keypts, counts = np.unique(all_keypts, axis=0, return_counts=True)
  #   # Get all duplicate elements from all_keypoints
  #   n = len(unique_keypts)
  #   corners = [ list(unique_keypts[i]) for i in range(n) if counts[i] > 1 ]

  #   # Remove all corners from the sources and sinks 
  #   for source in sources:
  #     if source[0:2] in corners:
  #       sources.remove(source)

  #   for sink in sinks:
  #     if sink[0:2] in corners:
  #       sinks.remove(sink)

  return np.array(sources), np.array(sinks)

def write(filename, sources, sinks, tau, diff_sqrt):

    with open(filename, "w") as config_file:
        config_file.write("sources:\n")
        for y, x, theta in sources:
            config_file.write("  - {y: %d, x: %d, theta: %d}\n"%(y, x, theta))

        config_file.write("sinks:\n")
        for y, x, theta in sinks:
            config_file.write("  - {y: %d, x: %d, theta: %d}\n"%(y, x, theta))
    
        config_file.write(f"""
plot_source_fields: False
plot_sink_fields: False
plot_completion_fields: True

save_source_data: False
save_sink_data: False
save_completion_data: True

seed: 0
width: {img_w}
diff_sqrt: {diff_sqrt}
tau: {tau}

#INPUT/OUTPUT FILES
source_file: source_file
source_img_file: source_img
source_field_data: source_field_data
sink_file: sink_file
sink_img_file: sink_img
sink_field_data: sink_field_data
completion_file: completion_field
completion_img_file: completion_img
completion_field_data: completion_field_data
source_sink_sum_file: source_sink_sum_file
source_sink_sum_img_file: source_sink_sum_img
        """)
  

def main(taus, diffs, img_w, configs_directory, sandvld_directory, single_linedrawing=False, linedrawing_num=0):
  line_drawings_middle = loadmat(f'{sandvld_directory}/SandVLDs_middle_equalLength.mat')
  line_drawings_junctions = loadmat(f'{sandvld_directory}/SandVLDs_junction_equalLength.mat')
  line_drawings_full = loadmat(f'{sandvld_directory}/SandVLDs_cleaned.mat')
  LD_middle = line_drawings_middle["middleLD"]
  LD_junction = line_drawings_junctions["junctionLD"]
  LD_full = line_drawings_full["LD"]

  num_linedrawings = trange(linedrawing_num, linedrawing_num + 1)
  if not single_linedrawing:
    num_linedrawings = trange(len(LD_middle[0]))

  for i in num_linedrawings:
    for tau in taus:
      for diff in diffs:

        ld_folder = f"{configs_directory}/ld_{i}"
        create_path(ld_folder, directory=True)

        diff_sqrt = np.sqrt(diff)
        figure_middle = LD_middle[0][i]
        strokes_middle = figure_middle[2][0]
        sources_mid, sinks_mid = get_sources_and_sinks(strokes_middle, junctions=False)

        create_path(f"{ld_folder}/middle", directory=True)
        middle_filename = f"{ld_folder}/middle/ld_middle_{i}_tau_{tau}_diff_{np.round(diff, 3)}.yml"
        write(middle_filename, sources_mid, sinks_mid, tau, diff_sqrt)

        figure_junction = LD_junction[0][i]
        strokes_junction = figure_junction[2][0]
        sources_junc, sinks_junc = get_sources_and_sinks(strokes_junction, junctions=True)

        create_path(f"{ld_folder}/junc", directory=True)
        junc_filename = f"{ld_folder}/junc/ld_junc_{i}_tau_{tau}_diff_{np.round(diff, 3)}.yml"
        write(junc_filename, sources_junc, sinks_junc, tau, diff_sqrt)

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description="")
  parser.add_argument('--configs_directory', type=str, required=True)
  parser.add_argument('--sandvld_directory', type=str, required=True)
  parser.add_argument('--line_drawing', type=int, required=False)
  parser.add_argument('--tau', type=int, required=False)
  parser.add_argument('--diff', type=float, required=False)
  args = parser.parse_args()

  # NOTE: We divided the width, and all coordinates by 8 to reduce computation time
  img_w = 256

  create_path(args.configs_directory, directory=True)

  if args.line_drawing is not None:
    main([args.tau], [args.diff], img_w, args.configs_directory, args.sandvld_directory, single_linedrawing=True, linedrawing_num=args.line_drawing)
  else:
    diffs = [0.001] + [0.005 * i for i in range(1, 7)]
    taus = [1] + [1 * i for i in range(10, 160, 10)]
    main(taus, diffs, img_w, args.configs_directory, args.sandvld_directory)
