import random
import numpy as np
import utils
import os
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

x_dim, y_dim = 64, 64
max_height = 800
max_s_options = [0.3 * x_dim, 0.25 * x_dim, 0.2 * x_dim, 0.15 * x_dim, 0.225 * x_dim]


def get_random_parameters(x_dim, y_dim, max_height):
    bumps = []
    max_s = random.choice(max_s_options)
    length = random.choice([1, 2, 3, 4, 5])

    for _ in range(length):
        height = random.random() * 2 / 3 * max_height + 1 / 3 * max_height
        s = random.random() * 2 / 3 * max_s + max_s / 3
        x = random.random() * x_dim if random.random() > 0.5 else random.random() * x_dim * -1
        y = random.random() * y_dim if random.random() > 0.5 else random.random() * y_dim * -1
        center = (x, y, 0)
        bump = (center, height, s)
        bumps.append(bump)

    return bumps


def generate_data(data_points, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

    for i in tqdm(range(data_points)):
        bumps = get_random_parameters(x_dim, y_dim, max_height)
        grid = utils.generate_complex_bumps(x_dim, y_dim, bumps, step=1)
        color_profile = utils.generate_colors(grid, step=1)

        # Normalize the color profile to the range [0, 255] and convert to uint8
        normalized_color_profile = (color_profile - color_profile.min()) / (color_profile.max() - color_profile.min())
        image_data = (normalized_color_profile * 255).astype(np.uint8)

        # Create an image from the array and save it as a PNG file
        img = Image.fromarray(image_data)
        file_name = f"{folder}/color_{i + 1}.png"
        img.save(file_name)

bumps = get_random_parameters(x_dim, y_dim, max_height)
grid = utils.generate_complex_bumps(x_dim, y_dim, bumps, step=1)
color_profile = utils.generate_colors(grid, step=1)

# Normalize the color profile to the range [0, 255] and convert to uint8
normalized_color_profile = (color_profile - color_profile.min()) / (color_profile.max() - color_profile.min())
image_data = (normalized_color_profile * 255).astype(np.uint8)

# Create an image from the array and save it as a PNG file
img = Image.fromarray(image_data)
plt.imshow(img)
plt.show()

def peek(path):
    try:
        img = Image.open(path)
        plt.imshow(img)
        plt.tight_layout()
        plt.title("Clean Image")
        plt.show()

        img_array = np.array(img)
        img_array = img_array.astype(np.float32)

        noise = np.random.normal(0, 6, img_array.shape)
        noisy_img = img_array + noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

        squared_difference = (img_array - noisy_img) ** 2
        print(np.mean(squared_difference))

        plt.imshow(noisy_img)
        plt.title("Noisy Image")
        plt.tight_layout()
        plt.show()

        normalized_squared_difference = squared_difference / squared_difference.max()

        plt.imshow(normalized_squared_difference)
        plt.title("Squared Difference")
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"Error: File ot found.")


generate_data(200, "test_blue")
peek("test_blue/color_4.png")
