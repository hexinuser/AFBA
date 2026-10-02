# AFBA

This repository contains the Python code accompanying the paper **“From inertial dynamics with implicit Hessian damping to accelerated forward–backward algorithms”** by **Xin He and Ya-Ping Fang**.

The code compares AFBA, which uses distinct extrapolation and gradient-evaluation coefficients, with a matched FISTA method on a wavelet-based image deblurring problem. The experiments use the  Cameraman image to illustrate the theoretical results and compare iteration counts, objective values, and PSNR.

The implementation requires Python, NumPy, Pillow, and Matplotlib. Place `cameraman.png` in the same folder as `compare_cameraman.py` and run the script. Parameters can be adjusted at the top of the file. Tables and restored images are saved in `outputs`.
