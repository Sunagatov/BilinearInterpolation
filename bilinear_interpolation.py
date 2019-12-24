import os
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import imageio
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule


def get_bilinear_pixel(image: np.ndarray, x: int, y: int, scale: int = 2) -> list:
    out = []

    position_x = x / scale
    position_y = y / scale
    mod_xi = int(position_x)
    mod_yi = int(position_y)
    mod_xf = position_x - mod_xi
    mod_yf = position_y - mod_yi
    mod_xi_plus_one_lim = 0 if mod_xi + 1 >= image.shape[0] else mod_xi + 1
    mod_yi_plus_one_lim = 0 if mod_yi + 1 >= image.shape[1] else mod_yi + 1

    for chan in range(image.shape[2]):
        bl = image[mod_xi, mod_yi, chan]
        br = image[mod_xi_plus_one_lim, mod_yi, chan]
        tl = image[mod_xi, mod_yi_plus_one_lim, chan]
        tr = image[mod_xi_plus_one_lim, mod_yi_plus_one_lim, chan]

        b = (1.0 - mod_xf) * bl + mod_xf * br
        t = (1.0 - mod_xf) * tl + mod_xf * tr
        pxf = (1.0 - mod_yf) * b + mod_yf * t
        out.append(int(pxf))

    return out


def get_image_enlarged_shape(image: np.ndarray, scale: int = 2) -> list:
    return list(map(int, [image.shape[0] * scale, image.shape[1] * scale, image.shape[2]]))


def interpolate_image_manually(image: np.ndarray, scale: int = 2):
    start = time.time()

    print('Getting image enlarged shape..')
    enlarged_image_shape = get_image_enlarged_shape(image, scale)
    enlarged_image = np.empty(enlarged_image_shape, dtype=np.uint8)

    print('Calculating bilinear pixels..')
    for x in range(enlarged_image.shape[0]):
        for y in range(enlarged_image.shape[1]):
            enlarged_image[x, y] = get_bilinear_pixel(image, x, y, scale)

    print(f'interpolate_image_manually - calculation time: {time.time() - start:.5f} s')
    return enlarged_image


def interpolate_image_by_cuda(image: np.ndarray, scale: int = 2):
    import pycuda.autoinit

    cu_module = SourceModule(open("kernel.cu", "r").read())
    interpolate = cu_module.get_function("interpolate")

    start = time.time()

    print('Getting color channels..')
    uint32_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint32)
    for x in range(uint32_image.shape[0]):
        for y in range(uint32_image.shape[1]):
            for ch in range(image.shape[2]):
                uint32_image[x, y] += image[x, y, ch] << (8 * (image.shape[2] - ch - 1))

    print('Copying texture..')
    cu_tex = cu_module.get_texref("tex")
    cu_tex.set_filter_mode(cuda.filter_mode.POINT)
    cu_tex.set_address_mode(0, cuda.address_mode.CLAMP)
    cu_tex.set_address_mode(1, cuda.address_mode.CLAMP)
    cuda.matrix_to_texref(uint32_image, cu_tex, order="C")

    print('Getting image enlarged shape..')
    enlarged_image_shape = get_image_enlarged_shape(image, scale)
    result = np.zeros((enlarged_image_shape[0], enlarged_image_shape[1]), dtype=np.uint32)
    block = (16, 16, 1)
    grid = (
        int(np.ceil(enlarged_image_shape[0] / block[0])),
        int(np.ceil(enlarged_image_shape[1] / block[1]))
    )

    print('Interpolating..')
    interpolate(cuda.Out(result),
                np.int32(image.shape[1]),
                np.int32(image.shape[0]),
                np.int32(enlarged_image_shape[1]),
                np.int32(enlarged_image_shape[0]),
                np.int32(image.shape[2]),
                block=block,
                grid=grid,
                texrefs=[cu_tex])

    print('Combining channels into color points..')
    rgba_image = np.zeros((enlarged_image_shape[0], enlarged_image_shape[1], image.shape[2]), dtype=np.uint32)
    for x in range(rgba_image.shape[0]):
        for y in range(rgba_image.shape[1]):
            output_x_y = result[x, y]
            for ch in range(rgba_image.shape[2]):
                rgba_image[x, y, rgba_image.shape[2] - ch - 1] = output_x_y % 256
                output_x_y >>= 8

    print('Clearing temporaries..')
    del result
    del uint32_image
    print(f'interpolate_image_by_cuda - calculation time: {time.time() - start:.5f} s')
    return rgba_image


def show_image(image: np.ndarray, title: str = ''):
    plt.imshow(image)
    plt.title(title)
    plt.show()


def calculate_image(file: str, scale: int = 2):
    print(f'Image {file} - calculation started..')
    image = imageio.imread(file)
    image_by_cuda = interpolate_image_by_cuda(image, scale)
    image_interpolated_manually = interpolate_image_manually(image, scale)
    print('Comparing results..')
    np.testing.assert_array_equal(image_by_cuda, image_interpolated_manually)
    print('Showing images..')
    show_image(image, 'Original image')
    show_image(image_by_cuda, 'CUDA interpolation')
    show_image(image_interpolated_manually, 'Manual interpolation')
    print(f'Image {file} - calculation finished!')


def get_images_files(folder: str = './data') -> List[str]:
    # images_files = [
    #     f'{folder}/ezh.png'
    # ]

    images_files = []
    for ignoredRoot, ignoredDirs, files in os.walk(folder):
        images_files += [f'{folder}/{f}' for f in files if f.endswith(".png") and not f.startswith("ignored")]

    return images_files


if __name__ == "__main__":
    folder = './data'
    for file in get_images_files(folder):
        calculate_image(file)
