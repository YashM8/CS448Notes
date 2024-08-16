"""
These data generating functions are taken from Frostad Research Group.
"""

import numpy as np
from math import floor, ceil
import matplotlib.pyplot as plt
import csv


def generate_bump(x_dim, y_dim, stats, center=np.array([0, 0, 0]), step=1):
    height, s = stats
    x_min, x_max = floor(-x_dim / step), ceil(x_dim / step + 1)
    y_min, y_max = floor(-y_dim / step), ceil(y_dim / step + 1)

    def get_z(x, y):
        return height * np.exp(-1 * (((x - center[0]) / s) ** 2 + ((y - center[1]) / s) ** 2))

    grid_points = [[get_z(x * step, y * step) for y in range(y_min, y_max)] for x in
                   range(x_min, x_max)]  # rows are x, columns are y
    return np.array(grid_points)


# generates complex bump structure by summing individual bumps
# bumps must be passed as [((x, y, z), height, s), ...]
def generate_complex_bumps(x_dim, y_dim, bumps, step=1):
    x_min, x_max = floor(-x_dim / step), ceil(x_dim / step + 1)
    y_min, y_max = floor(-y_dim / step), ceil(y_dim / step + 1)
    grid_points = np.zeros((x_max - x_min, y_max - y_min))

    for b in range(len(bumps)):
        grid_points += generate_bump(x_dim, y_dim, (bumps[b][1], bumps[b][2]), np.array(bumps[b][0]),
                                     step=step)  # (bumps[b][0][0], bumps[b][0][1])
    grid_points = grid_points[1:, :-1]

    return grid_points


def display_3d(thickness, color, step=1, save_name=None):
    # colors = np.reshape(color, (color.shape[0] * color.shape[1], 3)) / 255.

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection='3d')
    xs, ys, zs = [], [], []
    x_shift = floor(thickness.shape[0] / 2)
    y_shift = floor(thickness.shape[1] / 2)
    for x in range(thickness.shape[0]):
        for y in range(thickness.shape[1]):
            z = thickness[x, y]
            xs.append((x - x_shift) * step)
            ys.append((y - y_shift) * step)
            zs.append(z)

    ax.scatter(xs, ys, zs, cmap='inferno', c=zs)
    ax.set_zlim(0, 1000)
    ax.view_init(elev=40, azim=270)
    plt.tight_layout()
    if save_name: plt.savefig(f"{save_name}.png", pad_inches=0)
    plt.show()


def display_2d(color, step=1, title=""):
    xs, ys = [], []
    x_shift = floor(color.shape[0] / 2)
    y_shift = floor(color.shape[1] / 2)
    for x in range(color.shape[0]):
        for y in range(color.shape[1]):
            xs.append((x - x_shift) * step)
            ys.append((y - y_shift) * step)
    c = np.reshape(color, (color.shape[0] * color.shape[1], 3)) / 255.

    plt.scatter(xs, ys, color=c)
    plt.title(title)
    plt.show()


# Existing functions (created for analyzer app)
def get_intensity(h, I0, L, n1, n2, n3):
    """
    This function calculates the intensity of light from an interferometer as
    a function of film thickness, initial intensity, wavelenghth, and
    refractive indices.

    INPUT:
        h = 1-D numpy array of thickness values desired
        I0 = 1-D numpy array of light intensity at each wavelength L
        L = 1-D numpy array of relevant wavelength of the light source
        n1,n2,n3 = refractive indices of the first, second, and third layers of
                material that the light encounters during its path.
    OUTPUT:
        intensity = numpy array of size (len(L),len(h)) with the calculated
                intensities at each value of L and h.
    """

    # Reshape input arrays appropriately
    assert len(L) == len(I0)
    L = L.reshape((len(L), 1))
    I0 = I0.reshape((len(I0), 1))
    h = h.reshape((1, len(h)))
    # Compute the relative intensities due to reflection properties using
    # Fresnal equations
    R1 = ((n1 - n2) / (n1 + n2)) ** 2
    R2 = ((n2 - n3) / (n2 + n3)) ** 2
    # Compute the intensities
    A = 2 * np.sqrt(R1 * R2 * (1 - R1) ** 2) / (R1 + R2 * (1 - R1) ** 2)
    D = 4 * n2 * np.pi / L * h + np.pi * ((n2 > n1) + (n3 > n2))
    intensity = I0 * (1 + A * np.cos(D))
    #    intensity = I0*np.cos(D/2)*np.cos(D/2)
    intensity /= np.max(intensity)

    return intensity


# generates a color map of shape (h x 3) for what the camera would record for each thickness, given the setup specified in the
# optics file and by the refractive indices of the materials
def generate_color_map(h, n1, n2, n3, optics):
    I0 = optics["I0"]
    L = optics["wavelength"]
    I = get_intensity(h, I0, L, n1, n2, n3)
    I_f = np.multiply(I.T, optics["filter"]).T  # raw light going into the camera
    sensor = np.stack((optics["R_camera"], optics["G_camera"], optics["B_camera"])).T  # camera sensor sensitivities
    RGB = I_f.T @ sensor  # the colors actually recorded by the camera

    return RGB


def parse_optics_file():
    """
    Look for the file "optics.csv" in the code folder and then read this to
    get the information needed to generate a colormap.

    Returns a dictionary containing the data needed.
    """
    # Initialize list for storing file contents
    optics = []

    # Read file (skipping header) and store values in list of lists
    path = "optics.csv"
    with open(path, 'r') as f:
        temp = csv.reader(f)
        light_source = next(temp)[1]
        light_filter = next(temp)[1]
        camera_type = next(temp)[1]
        header = next(temp)
        assert len(header) == 6

        for row in temp:
            optics += [[float(x) for x in row]]
    # convert to numpy array
    optics = np.array(optics)
    # converty to dictionary
    temp = {}
    header = ['wavelength', 'I0', 'filter', 'R_camera', 'G_camera', 'B_camera']
    temp['light_source'] = light_source
    temp['filter_name'] = light_filter
    temp['camera'] = camera_type
    for i in range(len(header)):
        temp[header[i]] = (optics[:, i])

    optics = temp

    return optics


def generate_colors(thickness_profile, step=1):
    optics = parse_optics_file()
    h = np.array(range(0, ceil(np.max(thickness_profile) / step) + 1)) * step
    n1 = 1
    n2 = 1.333
    n3 = 1.5

    RGB = generate_color_map(h, n1, n2, n3, optics)
    RGB = RGB / np.max(RGB) * 255.  # scale values to between 0 and 255

    color_profile = RGB[
        np.where(np.round(thickness_profile / step).astype(int) > 0, np.round(thickness_profile / step).astype(int), 0)]

    return color_profile


def show_images(original, noisy, denoised):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title('Original Image S1')
    axes[1].imshow(noisy.permute(1, 2, 0).cpu().numpy())
    axes[1].set_title('Noisy Image S1')
    axes[2].imshow(denoised.permute(1, 2, 0).cpu().numpy())
    axes[2].set_title('Denoised Image S1')
    for ax in axes:
        ax.axis('off')
    plt.show()
