import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os
import json

RADIANS_PER_ROTATION = 2 * np.pi
DEGREES_PER_RADIAN = 180 / np.pi

def _normalize_angle(theta):
    # compute the equivalent signed principle angle.
    theta = theta % RADIANS_PER_ROTATION

    # compute the equivalent absolute principle angle.
    theta = (theta + RADIANS_PER_ROTATION) % RADIANS_PER_ROTATION

    if theta > RADIANS_PER_ROTATION / 2:
        theta -= RADIANS_PER_ROTATION

    return theta


def _rotate_by_random_angle(theta, diffusivity=0.05):
    # the signed angle by which to rotate.
    dtheta = np.random.normal(0, np.sqrt(diffusivity))

    # compute the new signed orientation.
    theta += dtheta

    # compute the normalized angle.
    theta = _normalize_angle(theta)

    return theta


def _take_step(x0, y0, theta):
    return x0 + np.cos(theta), y0 + np.sin(theta)


def _random_walk(origin=(0, 0, 0), T=1000, tau=20):
    states = [origin]
    x, y, theta = origin

    for t in range(0, T):
        theta = _rotate_by_random_angle(theta)
        x, y = _take_step(x, y, theta)

        states.append((x, y, theta))

        if (np.random.binomial(1, 2 ** (- 1 / tau)) == 0):
            break

    return states


def random_walks(n, origin=(0, 0, 0), T=1000, tau=20):
    walks = []
    for i in tqdm(range(n)):
        walks.append(_random_walk(origin, T, tau))

    return walks


def _discretize_walk(states, w=256, b=36):
    # determine the bin size.
    bin_size = int(RADIANS_PER_ROTATION * DEGREES_PER_RADIAN / b)

    hw = int(w / 2)
    hb = int(b / 2)

    m = np.zeros((w, w, b))

    for state in states:
        x, y, theta = state

        x = int(np.floor(x))
        y = int(np.floor(y))

        # ignore the state if the position is beyond the neighbourhood.
        if x < -hw or x > hw - 1 or y < -hw or y > hw - 1:
            continue

        bin = int((theta * DEGREES_PER_RADIAN / bin_size) + hb)

        # if (np.all(m[y + hw, x + hw, :] == 0)):
        m[y + hw, x + hw, bin] = 1
    return m


def discretize_walks(walks, w=256, b=36):
    m = np.zeros((w, w, b))

    for walk in tqdm(walks):
        m += _discretize_walk(walk, w, b)

    return m


def rotate_and_translate(walks, dx, dy, dtheta, origin):

    x0, y0 = origin

    new_walks = []
    for walk in walks:

        new_walks.append([])

        for (x, y, theta) in walk:

            nx = x0 + np.cos(dtheta) * (x - x0) - \
                np.sin(dtheta) * (y - y0) + dx
            ny = y0 + np.sin(dtheta) * (x - x0) + \
                np.cos(dtheta) * (y - y0) + dy

            theta += dtheta
            theta = _normalize_angle(theta)

            new_walks[-1].append((nx, ny, theta))

    return new_walks


def matrix_point(s):
    x, y, theta = s
    return (y + 128, x + 128, int(theta // 10) + 18)

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