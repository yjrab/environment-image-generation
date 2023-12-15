import cv2
import numpy as np
import itertools
import os

# choose either "prod" or "testing"
phase = "testing"
# choose either "png" or "hdr"
mode = "png"
# choose either "map_black" or "map" ("map-min" for compressed version)
map_image = "map_black"
# choose either 8 or 4 or 2 or 1
divider = 4
# choose either square or diffuse or specular
light_type = "specular"

base_path = "./base_envs" + "_" + mode
generated_path = "./envs_generated" + "_" + mode
testing_path = "./testing"

image_main = cv2.imread(os.path.join(base_path, map_image + "." + mode))
image_size = image_main.shape[0]
filename = "env"

block_size = int(image_size / divider)
blocks_ver_min = 1
blocks_hor_min = 1
blocks_ver_max = image_size // block_size
blocks_hor_max = image_size // block_size

grid = {
    (0, 0): "A0",
    (0, 256): "A1",
    (0, 512): "A2",
    (0, 768): "A3",
    (0, 1024): "A4",
    (0, 1280): "A5",
    (0, 1536): "A6",
    (0, 1792): "A7",
    (256, 0): "B0",
    (256, 256): "B1",
    (256, 512): "B2",
    (256, 768): "B3",
    (256, 1024): "B4",
    (256, 1280): "B5",
    (256, 1536): "B6",
    (256, 1792): "B7",
    (512, 0): "C0",
    (512, 256): "C1",
    (512, 512): "C2",
    (512, 768): "C3",
    (512, 1024): "C4",
    (512, 1280): "C5",
    (512, 1536): "C6",
    (512, 1792): "C7",
    (768, 0): "D0",
    (768, 256): "D1",
    (768, 512): "D2",
    (768, 768): "D3",
    (768, 1024): "D4",
    (768, 1280): "D5",
    (768, 1536): "D6",
    (768, 1792): "D7",
    (1024, 0): "E0",
    (1024, 256): "E1",
    (1024, 512): "E2",
    (1024, 768): "E3",
    (1024, 1024): "E4",
    (1024, 1280): "E5",
    (1024, 1536): "E6",
    (1024, 1792): "E7",
    (1280, 0): "F0",
    (1280, 256): "F1",
    (1280, 512): "F2",
    (1280, 768): "F3",
    (1280, 1024): "F4",
    (1280, 1280): "F5",
    (1280, 1536): "F6",
    (1280, 1792): "F7",
    (1536, 0): "G0",
    (1536, 256): "G1",
    (1536, 512): "G2",
    (1536, 768): "G3",
    (1536, 1024): "G4",
    (1536, 1280): "G5",
    (1536, 1536): "G6",
    (1536, 1792): "G7",
    (1792, 0): "H0",
    (1792, 256): "H1",
    (1792, 512): "H2",
    (1792, 768): "H3",
    (1792, 1024): "H4",
    (1792, 1280): "H5",
    (1792, 1536): "H6",
    (1792, 1792): "H7",
}


def grid_conv(row, col=0):
    return grid[(row, col)]


def add_square_light(pos_ver, pos_hor, block_size, image):
    image[pos_ver : pos_ver + block_size, pos_hor : pos_hor + block_size] = 255


def add_diffuse_light(pos_ver, pos_hor, block_size, image):
    center = (pos_ver + (block_size // 2), pos_hor + (block_size // 2))
    radius = block_size // 2
    y, x = np.ogrid[pos_ver : pos_ver + block_size, pos_hor : pos_hor + block_size]

    circle_mask = (y - center[0]) ** 2 + (x - center[1]) ** 2 <= radius**2
    circle_mask = circle_mask.astype(int) * 255
    circle_mask = np.tile(circle_mask[:, :, np.newaxis], 3)
    image[pos_ver : pos_ver + block_size, pos_hor : pos_hor + block_size] = circle_mask


def add_specular_light(pos_ver, pos_hor, block_size, image):
    center = (pos_ver + (block_size // 2), pos_hor + (block_size // 2))
    radius = block_size // 2
    y, x = np.ogrid[pos_ver : pos_ver + block_size, pos_hor : pos_hor + block_size]

    circle_mask = ((y - center[0]) ** 2 + (x - center[1]) ** 2) / radius**2
    circle_mask = 1 - circle_mask
    circle_mask = np.clip(circle_mask, 0, 1) * 255
    circle_mask = np.tile(circle_mask[:, :, np.newaxis], 3)
    image[pos_ver : pos_ver + block_size, pos_hor : pos_hor + block_size] = circle_mask


def save_image(mode, path, image, filename):
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if mode == "png":
        cv2.imwrite(
            os.path.join(path, filename + "." + mode),
            image,
            [int(cv2.IMWRITE_PNG_COMPRESSION), 4],
        )
    elif mode == "hdr":
        cv2.imwrite(os.path.join(path, filename + "." + mode), image)


def generate_envs(
    image_size,
    block_size,
    blocks_ver_min,
    blocks_ver_max,
    blocks_hor_min,
    blocks_hor_max,
):
    positions_ver = range(0, image_size - block_size + 1, block_size)

    for blocks_ver in range(blocks_ver_min, blocks_ver_max + 1):
        print(f"blocks_ver: {blocks_ver} / {blocks_ver_max}")
        combinations = list(itertools.combinations(positions_ver, blocks_ver))

        for combo in combinations:
            for blocks_hor in range(blocks_hor_min, blocks_hor_max + 1):
                image = image_main.copy()

                for pos_ver in combo[:blocks_ver]:
                    for light in range(blocks_hor):
                        if light_type == "square":
                            add_square_light(
                                pos_ver,
                                (image_size // blocks_hor) * light,
                                block_size,
                                image,
                            )
                        elif light_type == "diffuse":
                            add_diffuse_light(
                                pos_ver,
                                (image_size // blocks_hor) * light,
                                block_size,
                                image,
                            )
                        elif light_type == "specular":
                            add_specular_light(
                                pos_ver,
                                (image_size // blocks_hor) * light,
                                block_size,
                                image,
                            )

                filename_variant = (
                    filename
                    + f"_{block_size//256}blocksize_{blocks_ver}verlights_{blocks_hor}horlights_{light_type}light_{'+'.join(map(grid_conv, combo))}"
                )
                if phase == "prod":
                    save_image(mode, generated_path, image, filename_variant)
                elif phase == "testing":
                    save_image(mode, testing_path, image, filename_variant)


generate_envs(
    image_size,
    block_size,
    blocks_ver_min,
    blocks_ver_max,
    blocks_hor_min,
    blocks_hor_max,
)
