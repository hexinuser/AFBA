"""Compare AFBA and a matched FISTA-type method on cameraman.png.
Dependencies: numpy, Pillow, matplotlib.
"""
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Parameters: edit this section to change the experiment.
IMAGE_FILE = 'cameraman.png'  # Input path relative to this script; expected size: 264 x 264.
ALPHAS = [4, 6, 10]          # Inertial parameters, greater than 3.
THETAS = [0, 0.5, 0.95]     # 0: matched FISTA-type method; positive values: AFBA.
TOLERANCES = [1e-4, 1e-5, 1e-6]  # Normalized proximal-gradient stopping tolerances.
SL = 0.95                  # Step size s = SL/L.
MU = 1e-3                  # Haar l1 regularization weight.
NOISE_SIGMA = 1e-3          # Gaussian noise standard deviation.
SEED = 10000           # Fixed seed; all methods use the same observation.

BLUR_SIZE = 9              # Odd Gaussian kernel width.
BLUR_SIGMA = 1.5           # Gaussian blur standard deviation.
LEVELS = 3                # Number of orthonormal Haar decomposition levels.
MAX_ITER = 10000           # Maximum number of updates per parameter pair.
TABLE_ALPHA = 6            # Alpha used for Table 2 and the comparison figure.
DISPLAY_THETA = 0.95       # AFBA theta displayed beside FISTA-type.

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'outputs'


def load_image():
    """Read the local PNG at its original size and normalize to [0, 1]."""
    with Image.open(ROOT / IMAGE_FILE) as picture:
        # Preserve 16-bit grayscale inputs; convert color inputs to grayscale.
        if picture.mode in ('I', 'I;16', 'I;16B', 'I;16L'):
            x = np.asarray(picture, dtype=np.float64) / 65535
        else:
            x = np.asarray(picture.convert('L'), dtype=np.float64) / 255
    return x


def make_otf(n):
    """Build the Fourier multiplier of the normalized periodic Gaussian blur."""
    r = np.arange(BLUR_SIZE) - BLUR_SIZE // 2
    kernel = np.exp(-(r[:, None]**2 + r[None, :]**2) / (2 * BLUR_SIGMA**2))
    padded = np.zeros((n, n))
    padded[:BLUR_SIZE, :BLUR_SIZE] = kernel / kernel.sum()
    # Move the kernel center to the Fourier origin.
    padded = np.roll(padded, (-(BLUR_SIZE // 2),) * 2, axis=(0, 1))
    return np.fft.fft2(padded)


def blur(x, otf):
    """Apply periodic convolution using the Fourier multiplier."""
    return np.fft.ifft2(otf * np.fft.fft2(x)).real


def haar(x, inverse=False):
    """Apply the orthonormal 2D Haar transform or its inverse."""
    w = x.copy()
    levels = range(LEVELS - 1, -1, -1) if inverse else range(LEVELS)
    for level in levels:
        size = x.shape[0] // 2**level
        half = size // 2
        a = w[:size, :size].copy()
        if inverse:
            # Reconstruct rows, then columns, from coarse to fine scales.
            temp = np.empty_like(a)
            temp[0::2] = (a[:half] + a[half:]) / np.sqrt(2)
            temp[1::2] = (a[:half] - a[half:]) / np.sqrt(2)
            a[:, 0::2] = (temp[:, :half] + temp[:, half:]) / np.sqrt(2)
            a[:, 1::2] = (temp[:, :half] - temp[:, half:]) / np.sqrt(2)
        else:
            # Separate averages and differences in columns, then rows.
            a = np.concatenate((a[:, 0::2] + a[:, 1::2],
                                a[:, 0::2] - a[:, 1::2]), axis=1) / np.sqrt(2)
            a = np.concatenate((a[0::2] + a[1::2],
                                a[0::2] - a[1::2]), axis=0) / np.sqrt(2)
        w[:size, :size] = a
    return w


def proximal(x, step):
    """Evaluate prox of step * MU * ||W x||_1 by Haar soft thresholding."""
    w = haar(x)
    w = np.sign(w) * np.maximum(np.abs(w) - step * MU, 0)
    return haar(w, inverse=True)


def gradient(x, otf, b_fft):
    """Compute 2 A*(A x-b), reusing the Fourier transform of b."""
    return 2 * np.fft.ifft2(otf.conj() * (otf * np.fft.fft2(x) - b_fft)).real


def compute_psnr(x, truth):
    """Compute unit-range PSNR using the unclipped restored image."""
    mse = np.mean((x - truth)**2)
    return np.inf if mse == 0 else -10 * np.log10(mse)


def run_method(b, truth, otf, alpha, theta):
    """Record the first stopping point for each tolerance and return the final image.
    Columns: alpha, theta, tolerance, iterations, objective, residual,
    relative_step, PSNR.
    """
    L = 2 * np.max(np.abs(otf)**2)
    step = SL / L
    b_fft = np.fft.fft2(b)
    x, previous = b.copy(), b.copy()
    records = np.zeros((len(TOLERANCES), 8))
    done = np.zeros(len(TOLERANCES), dtype=bool)
    for k in range(1, MAX_ITER + 1):
        # Use distinct extrapolation and gradient-evaluation coefficients.
        a = (k - 1) / (k + alpha - 1)
        d = (alpha - 3) * (2*k + alpha - 1) / (SL * (k + alpha - 1)**2)
        beta = a * (1 - theta * min(1, d))
        y = x + a*(x - previous)
        z = x + beta*(x - previous)
        new = proximal(y - step * gradient(z, otf, b_fft), step)
        relative_step = np.sqrt(np.sum((new - x)**2)) / max(1, np.sqrt(np.sum(x*x)))
        previous, x = x, new
        
        # Test the normalized proximal-gradient mapping with reference step 1/L.
        mapping = L * (x - proximal(x - gradient(x, otf, b_fft)/L, 1/L))
        residual = np.sqrt(np.sum(mapping*mapping)) / max(1, np.sqrt(np.sum(x*x)))
        for j, tolerance in enumerate(TOLERANCES):
            if not done[j] and (residual <= tolerance or k == MAX_ITER):
                objective = np.sum((blur(x, otf) - b)**2) + MU*np.abs(haar(x)).sum()
                psnr = compute_psnr(x, truth)
                records[j] = [alpha, theta, tolerance, k, objective, residual,
                              relative_step, psnr]
                done[j] = True
        if done.all():
            break
    return records, x


def save_table(name, header, rows):
    """Save a paper table as CSV"""
    text = ' | '.join(header) + '\n' + '\n'.join(' | '.join(row) for row in rows)
    (OUTPUT / (name + '.csv')).write_text(
        ','.join(header) + '\n' + '\n'.join(','.join(row) for row in rows) + '\n')
    # (OUTPUT / (name + '.txt')).write_text(text + '\n', encoding='utf-8')
    # (OUTPUT / (name + '.tex')).write_text(
    #     '\n'.join(' & '.join(row) + r' \\' for row in rows) + '\n')
    print('\n' + name + '\n' + text, flush=True)


def output_tables(rows):
    """Output iteration and quality tables; * marks an unmet tolerance."""
    table1, table2 = [], []
    for alpha in ALPHAS:
        for theta in THETAS:
            selected = rows[(rows[:, 0] == alpha) & (rows[:, 1] == theta)]
            method = 'FISTA' if theta == 0 else 'AFBA'
            counts = [str(int(r[3])) + ('' if r[5] <= r[2] else '*') for r in selected]
            table1.append([method, f'{alpha:.10f}', f'{theta:.10f}'] + counts)

            # Table 2 uses TABLE_ALPHA and the smallest stopping tolerance.
            if alpha == TABLE_ALPHA:
                r = selected[selected[:, 2] == min(TOLERANCES)][0]
                count = str(int(r[3])) + ('' if r[5] <= r[2] else '*')
                table2.append([
                    method, f'{theta:.10f}', count,
                    f'{r[4]:.10f}', f'{r[7]:.5f}'
                ])

    save_table(
        'table1_iterations',
        ['Method', 'alpha', 'theta']
        + [f'eps={t:.10f}' for t in TOLERANCES],
        table1
    )
    save_table(
        'table2_quality',
        ['Method', 'theta', 'iterations', 'objective', 'PSNR'],
        table2
    )
    print(f'\nTable 2: alpha={TABLE_ALPHA:.10f}, '
          f'tolerance={min(TOLERANCES):.10f}.')
    print('* means the tolerance was not met before MAX_ITER.')
    

def save_figure(images):
    """Save original/degraded/restored images and a single comparison figure."""
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), constrained_layout=True)
    names = ['Original', 'Blurred_noisy', 'FISTA', 'AFBA']
    for ax, x, name in zip(axes, images, names):
        # Clip for display only; all reported metrics use unclipped iterates.
        display = np.clip(x, 0, 1)
        Image.fromarray(np.rint(255*display).astype(np.uint8)).save(OUTPUT / (name + '.png'))
        ax.imshow(display, cmap='gray', vmin=0, vmax=1)
        ax.set_title(name.replace('_', '/'))
        ax.axis('off')
    fig.savefig(OUTPUT / 'comparison_man.png', dpi=220)
    plt.close(fig)


def main():
    """Run the single-image experiment and save results under outputs_P1."""
    if not (ALPHAS and THETAS and TOLERANCES and all(a > 3 for a in ALPHAS)
            and all(0 <= t <= 1 for t in THETAS) and all(t > 0 for t in TOLERANCES)
            and 0 < SL <= 1 and MU >= 0 and NOISE_SIGMA >= 0 and MAX_ITER >= 1
            and 1 <= BLUR_SIZE <= 256 and BLUR_SIZE % 2 == 1 and BLUR_SIGMA > 0
            and 1 <= LEVELS <= 8 and TABLE_ALPHA in ALPHAS
            and 0 in THETAS and DISPLAY_THETA in THETAS and DISPLAY_THETA > 0):
        raise ValueError('Check the parameters at the top of the script.')
    truth = load_image()
    OUTPUT.mkdir(exist_ok=True)
    otf = make_otf(truth.shape[0])
    # Generate the shared observation directly; no external noise file is needed.
    noise = np.random.default_rng(SEED).normal(size=truth.shape)
    b = blur(truth, otf) + NOISE_SIGMA*noise
    images = [truth, b, None, None]
    results = []
    print(f'Input: {ROOT / IMAGE_FILE}; size: {truth.shape}', flush=True)
    for alpha in ALPHAS:
        for theta in THETAS:
            records, restored = run_method(b, truth, otf, alpha, theta)
            results.extend(records)
            print(f'alpha={alpha:.10f}, theta={theta:.10f}, updates={int(records[-1, 3])}',
                  flush=True)
            if alpha == TABLE_ALPHA and theta in (0, DISPLAY_THETA):
                images[2 if theta == 0 else 3] = restored
    rows = np.array(results)
    header = 'alpha,theta,tolerance,iterations,objective,residual,relative_step,PSNR'
    formats = ['%.10f']*3 + ['%d'] + ['%.10f']*4
    # np.savetxt(OUTPUT / 'results.csv', rows, delimiter=',', header=header,
    #            comments='', fmt=formats)
    output_tables(rows)
    save_figure(images)
    print(f'\nSaved tables and images to: {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
