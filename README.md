# Visualizing Wake Characteristics for Bluff Bodies with varying Surface Roughness

## Overview

This repository contains the data processing pipeline for our final lab project studying how surface roughness affects the wake structure and drag behavior of circular cylinders in a soap film flow facility. We tested 12 cylinder configurations across 3 diameters (5, 10, 15 mm nominal) and 4 sandpaper grits (100, 150, 220, 400 ppi) at Reynolds numbers ranging from approximately 1,600 to 5,500.

The code handles TIFF image image processing, velocity field fitting, drag coefficient estimation, and Dynamic Mode Decomposition (DMD) for vortex-shedding frequency extraction.

<!-- <video src="Figures/Videos/T8_Open_Lab_3_5_400.mp4?raw=true" controls width="720"></video> -->

https://github.com/user-attachments/assets/570a39dd-b4ba-4d32-9356-70b5ab2caf6d

<!-- 5_400.mp4, 10_100.mp4, 10_150.mp4, 10_400.mp4, 15_150.mp4, 15_400.mp4 -->

## Repository Structure

```
├── testing.py                  # Core Test class - image processing, wake fitting, DMD
├── interactive.ipynb           # Qt GUI for manually labeling pathlines on images
├── analysis_v4.ipynb           # Full analysis pipeline: plots, Cd vs Re, Strouhal numbers, etc.
├── Visual_Measurements.xlsx    # Hand-labeled pathline endpoints
└── Figures/                    # Output figures directory
```

<!-- Raw calibration and image files `T8_Open_Lab_{1,2,3}` are shared [here](). -->

<!-- Each lab day folder contains subfolders named `{diameter}_{grit}/` holding the raw `.tiff` burst frames and a `calibration/` subfolder with the ruler image used to compute the magnification factor $\alpha$ (mm/px). -->

## Method Summary

### Image Processing
Raw high-speed frames (200 FPS, 5000 μs exposure) are captured with a FLIR Blackfly S camera. A calibration ruler image is used to compute a magnification factor $\alpha$ (mm/px) on each test day. A maximum-difference projection across all frames produces a composite pathline image.

### Cylinder Detection
The Hough Gradient Method (`cv.HoughCircles`) locates the cylinder center and radius in the mean-intensity background frame, giving the true diameter $D$ with sandpaper included. Manual overrides in `cylinder_override` are used for cases where the automated detection fails.

### Velocity Estimation
Pathline endpoints are labeled manually using the interactive GUI. For each labeled segment, velocity is computed as:

$$
U = \frac{\alpha \ l}{t_{exposure}}
$$

### Wake Fitting
A far-wake model [^Blevins] is fit to the collected ($x$, $y$, $U$) data points via `scipy.optimize.curve_fit`:

$$
U(x,y)=U_\infty \left(1-1.2\left( C_d \frac{D}{x} \right)^{1/2} e^{\frac{-13y^2}{C_d D x}} \right)
$$

Free-stream velocity $U_\infty$ and drag coefficient $C_d$ are both free parameters. A generalized power-law wake model is also fit as a secondary comparison.

### DMD Analysis
DFT did not work due to noise from the laser or lights (at 75Hz).
[SPOD](https://pyspod.readthedocs.io/en/latest/) did not work for this application, but I am less familiar with the meta-parameters, so it is conceivable.

Dynamic Mode Decomposition is applied to the near-wake pixel-intensity frames to extract dominant vortex-shedding frequencies. The decomposition is repeated over several SVD ranks $r$. At each rank the mode with the smallest spatial L1-norm is selected as the candidate shedding mode. The Strouhal number is then computed as:

$$
St = \frac{f D}{U_\infty}
$$

## Key Results

Key output figures:
- `vel_field_grid.png` - 4×3 grid of mean velocity field heatmaps
- `wake_grid.png` - Composite pathline images overlaid with velocity isolines
- `Cd_Re.png` - Crag coefficient vs. Reynolds number with error bars, trend fit, and 95% CI
- `psd_comparison.png` - Power spectral density of different parts of video (light/laser noise $\approx 75$ Hz)
- `grit_{N}_St_Re.png` - Strouhal number scatter plots per grit value. Horizontal reference lines mark the canonical circular-cylinder Strouhal number [^text][p. 324] $St_0 \approx 0.21$ and its sub-harmonics $St_0 / 2$ and $St_0 / 4$.

| Quantity | Value |
|---|---|
| Reynolds number range     | ~1,600 - 5,500 |
| Drag coefficient trend    | Cd = $40.2 Re^{-0.391}$ |
| Median wake model $R^2$   | 0.840 |
| Strouhal number reference | $St_0 \approx 0.21$ (smooth cylinder) |

Surface roughness was found to increase skin friction drag rather than delay boundary layer separation at the Reynolds numbers achievable in the soap film facility.

<!-- ## Future Work
1. Try BOPDMD
1. Add more data. 
1. more from report
-->

## Setup

**Python 3.10+ recommended.**

Create virtual environment with venv, uv, or conda, etc:
```bash
python -m venv .venv
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Cylinder detection, image I/O |
| `numpy` / `scipy` | Numerical computation, curve fitting, FFT |
| `pandas` | Data management |
| `openpyxl` | Reading `Visual_Measurements.xlsx` |
| `matplotlib` | Plotting |
| `pydmd` | Dynamic Mode Decomposition |

## License

Huge thanks to [Blake Vennall](https://www.linkedin.com/in/blake-vennall-b23a97270/) and [Logan Woodcock](https://www.linkedin.com/in/logan-woodcock-4a9991291/) for their work in conception, lab testing, and interpretation.

This code was developed as part of an open-ended lab project in ME 30801 at Purdue University. Feel free to adapt it for educational purposes with attribution to Rohan Patel.

## References

[^Blevins]: R. D. Blevins, “Forces on and stability of a cylinder in a wake,” *Journal of Offshore Mechanics and Arctic Engineering*, vol. 127, no. 1, pp. 39–45, Mar. 2005, ISSN: 0892-7219. DOI: [10.1115/1.1854697](10.1115/1.1854697). eprint: [https://asmedigitalcollection.asme.org/offshoremechanics/article-pdf/127/1/39/5555682/39_1.pdf](https://asmedigitalcollection.asme.org/offshoremechanics/article-pdf/127/1/39/5555682/39_1.pdf). [Online]. Available: [https://doi.org/10.1115/1.1854697](https://doi.org/10.1115/1.1854697).

I believe Eq. (3) has typo in the paper. The exponential should be inside the parentheses. Conceptually, the wake velocity is `free stream * (1 - spread * deficit)`.

[^text]: R. W. Fox, A. T. McDonald, and J. W. Mitchell, *Fox and McDonald's Introduction to Fluid Mechanics*. Wiley, 2020.
