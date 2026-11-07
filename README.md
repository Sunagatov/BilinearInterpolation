# Bilinear Interpolation

A Python and CUDA experiment comparing CPU and GPU image enlargement with bilinear interpolation.

<p>
  <a href="https://buymeacoffee.com/lucky_1uck">
    <img src="https://img.shields.io/badge/Support_the_project-FFDD00?style=for-the-badge&amp;logo=buymeacoffee&amp;logoColor=black" alt="Support the project" width="260" height="40">
  </a>
</p>

## What it does

- Resizes images with a Python/NumPy implementation and a PyCUDA kernel.
- Prints timing information for both implementations.
- Checks CPU/GPU equality with `numpy.testing.assert_array_equal`.
- Displays the original image and both results with Matplotlib.

## Requirements

Python 3, NumPy, Matplotlib, ImageIO, PyCUDA, and an NVIDIA GPU with a compatible
CUDA toolkit/driver. The kernel uses legacy CUDA texture references, so the
toolkit must support those APIs. Dependencies are not version-pinned.

## Run

Install the Python dependencies in a virtual environment:

```bash
python -m pip install numpy matplotlib imageio pycuda
```

Place a **PNG** image in [data](data), then run from the repository root:

```bash
python bilinear_interpolation.py
```

The entry point scans only `.png` files whose names do not start with `ignored`.
The supplied samples are JPEGs, so the unchanged script finds no inputs until
you add a PNG. To process a particular supplied file explicitly:

```bash
python -c "from bilinear_interpolation import calculate_image; calculate_image('data/cat2.jpg', scale=2)"
```

## Code map

- [bilinear_interpolation.py](bilinear_interpolation.py): CPU reference, GPU orchestration, comparison and display.
- [kernel.cu](kernel.cu): interpolation kernel loaded at runtime.

PyCUDA is imported at module level, including when you use the CPU functions.
Keep the working directory at the repository root so `kernel.cu` can be found.

## ☕ Support the project

If this project helps you learn or build, you can support its development.

<p>
  <a href="https://buymeacoffee.com/lucky_1uck">
    <img src="https://img.shields.io/badge/Support_the_project-FFDD00?style=for-the-badge&amp;logo=buymeacoffee&amp;logoColor=black" alt="Support the project" width="260" height="40">
  </a>
</p>
