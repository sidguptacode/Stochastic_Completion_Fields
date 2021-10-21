import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from tqdm import trange, tqdm

# the number of radians per rotation.
RADIANS_PER_ROTATION = 2 * np.pi

# the default square root of the diffusivity of the random walks.
SIGMA_SQRT = np.sqrt(0.01)

# the default value of tau for the random walks.
TAU = 20

# the default integer lattice size.
w, h = 256, 256

def discretize_points(points, w = w, h = h, n_bins = 36):    
    points = np.copy(points)
    
    points[:, 2] /= (2 * np.pi / n_bins) # compute the angle in radians to the bin number.
    points = np.floor(points) # floor the points.
    
    points = points[np.where(np.abs(points[:,0]) < h)]
    points = points[np.where(np.abs(points[:,1]) < w)]
    
    return points.astype(int)
    
def setup_initial_conditions(P, points, b = 36):   
    for point in points:
      P[tuple(point)]= 1 / len(points)
    return P

def translate_pointset(pointsset):
    # Convert sset into a matrix of points
    # NOTE: We negate the angle here, because we found that our results needed this to be flipped on the horizontal axis #TODO: Do we still do this?
    pointlist = np.array([[point['y'], point['x'], point['theta']*np.pi/180] for point in pointsset]).astype(float)
    return pointlist

def compute_fokker_planck(points, w = w, h = h, n_bins = 36, tau = TAU, sigma_sqrt = SIGMA_SQRT, src=True):

    # discretize the key points.
    points = discretize_points(points)

    # generate the initial pdf from the the discretized points.
    P = np.zeros((h, w, n_bins))
    P = setup_initial_conditions(P, points)

    P = np.array(P)

    # the possible angles from 0 to 2 * pi.
    possible_angles = np.linspace(0, 2 * np.pi * (1 - 1 / n_bins) , n_bins)
    
    # find the cosine ratios for the possible angles.
    cosine_ratios = np.cos(possible_angles)
    # set the cosine ratios for certain angles to actually be zero.
    cosine_ratios[[9, 27]] = 0
    # tile ratios to fit the size of P.
    cosine_ratios = np.tile(cosine_ratios, (h, w, 1))
    
    # find the sine ratios for the possible angles.
    sine_ratios = np.sin(possible_angles)
    # set the sine ratio for at zero radians to actually be zero.
    sine_ratios[[0, 18]] = 0
    # tile ratios to fit the size of P.
    sine_ratios = np.tile(sine_ratios, (h, w, 1))

    num_steps = w

    P_culm = np.zeros(P.shape)

    for i in trange(num_steps):
        
        """
          Step 1
        """
        # left shifted P matrix.
        Pl = np.zeros(P.shape)
        Pl[:, :-1, :] = P[:, 1:, :]

        # right shifted P matrix.
        Pr = np.zeros(P.shape)
        Pr[:, 1:, :] = P[:, :-1, :]

        pos_x_deltas = np.subtract(P[:, :, 0:9], Pr[:, :, 0:9])
        pos_cos_prod = np.multiply(cosine_ratios[:, :, 0:9], pos_x_deltas)

        P[:, :, 0:9] = np.subtract(P[:, :, 0:9], pos_cos_prod)

        pos_x_deltas = np.subtract(P[:, :, 28:], Pr[:, :, 28:])
        pos_cos_prod = np.multiply(cosine_ratios[:, :, 28:], pos_x_deltas)
        P[:, :, 28:] = np.subtract(P[:, :, 28:], pos_cos_prod)
        
        neg_x_deltas = np.subtract(Pl[:, :, 10:27], P[:, :, 10:27])
        neg_cos_prod = np.multiply(cosine_ratios[:, :, 10:27], neg_x_deltas)
        P[:, :, 10:27] = np.subtract(P[:, :, 10:27], neg_cos_prod)

        """
          Step 2
        """
        # up shifted P matrix.
        Pu = np.zeros(P.shape)
        Pu[:-1, :, :] = P[1:, :, :]

        # down shifted P matrix.
        Pd = np.zeros(P.shape)
        Pd[1:, :, :] = P[:-1, :, :]

        # matrix: difference between myself and point below me in the matrix
        neg_y_deltas = np.subtract(Pd[:, :, 19:], P[:, :, 19:])
        neg_sin_prod = np.multiply(sine_ratios[:, :, 19:], neg_y_deltas)
        P[:, :, 19:] = np.subtract(P[:, :, 19:], neg_sin_prod)

        pos_y_deltas = np.subtract(P[:, :, 1:18], Pu[:, :, 1:18])
        pos_sin_prod = np.multiply(sine_ratios[:, :, 1:18], pos_y_deltas)
        P[:, :, 1:18] = np.subtract(P[:, :, 1:18], pos_sin_prod)
  
        """
          Step 3
        """
        # in shifted P matrix.
        Pi = np.roll(P, 1, axis = 2)

        # out shifted P matrix.
        Po = np.roll(P, -1, axis = 2)

        delta_theta = (np.pi / 18)
        lamb = sigma_sqrt ** 2 / (2 * (delta_theta ** 2))

        P = lamb * Pi + (1 - 2 * lamb) * P + lamb * Po

        """
          Step 4
        """
        P *= np.exp(- 1 / tau)
        P_culm += P

    if not src:
      P_culm = np.roll(P_culm, n_bins // 2, axis = 2)

    return P_culm
