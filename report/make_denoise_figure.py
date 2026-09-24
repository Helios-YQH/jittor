"""Generate the denoising before/after figure for the technical report.

The sample is a test-set cloud, so no reference surface is available. The figure
shows the full cloud for context, a thin cross-section before and after
filtering (the filter's effect on a surface patch is visible as the band
thinning), and the distribution of the applied displacements.
"""

import os
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIGDIR = os.path.join(HERE, "figures")

NOISY = os.path.join(ROOT, "noisy.npy")
RESULT = "results/shapenet/02691156/1d6afc44b053ab07941d71475449eb25/denoised.npy"
ARCHIVE = os.path.join(ROOT, "result.zip")

INK = "#1f2328"
NOISY_C = "#D55E00"     # Okabe-Ito vermillion
DENOISED_C = "#0072B2"  # Okabe-Ito blue

SLAB_Z = 0.0
SLAB_HALF = 0.006


def load():
    noisy = np.load(NOISY).astype(np.float64)
    with zipfile.ZipFile(ARCHIVE) as z:
        denoised = np.load(z.open(RESULT)).astype(np.float64)
    return noisy, denoised


def main():
    noisy, denoised = load()
    rng = np.random.default_rng(0)
    sel = rng.choice(len(noisy), 9000, replace=False)

    slab = np.abs(noisy[:, 2] - SLAB_Z) < SLAB_HALF
    lo, hi = np.array([-0.9, -0.45]), np.array([0.9, 0.45])

    fig = plt.figure(figsize=(7.2, 2.3))
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    ax2 = fig.add_subplot(1, 4, 2)
    ax3 = fig.add_subplot(1, 4, 3)
    ax4 = fig.add_subplot(1, 4, 4)

    ax1.scatter(noisy[sel, 0], noisy[sel, 1], noisy[sel, 2], s=0.25, c=NOISY_C,
                alpha=0.5, linewidths=0, depthshade=False)
    ax1.set_xlim(-1, 1); ax1.set_ylim(-1, 1); ax1.set_zlim(-1, 1)
    ax1.set_box_aspect((1, 1, 1))
    ax1.view_init(elev=16, azim=-58)
    ax1.set_axis_off()
    ax1.set_title("(a) Noisy input", fontsize=8.5)

    for ax, pts, color, title in (
        (ax2, noisy[slab], NOISY_C, "(b) Cross-section, noisy"),
        (ax3, denoised[slab], DENOISED_C, "(c) Cross-section, denoised"),
    ):
        ax.scatter(pts[:, 0], pts[:, 1], s=1.6, c=color, alpha=0.7, linewidths=0)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        ax.grid(alpha=0.2, lw=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    ax2.set_xlabel("x", fontsize=8)
    ax3.set_xlabel("x", fontsize=8)
    ax2.set_ylabel("y", fontsize=8)

    disp = np.linalg.norm(denoised - noisy, axis=1)
    ax4.hist(disp, bins=70, color=DENOISED_C, alpha=0.9, lw=0)
    ax4.axvline(disp.mean(), color=INK, lw=1.0, ls="--")
    ax4.annotate(f"mean {disp.mean():.4f}", xy=(disp.mean(), 0),
                 xytext=(disp.mean() * 1.1, ax4.get_ylim()[1] * 0.5),
                 fontsize=7.5, color=INK,
                 arrowprops=dict(arrowstyle="->", lw=0.7, color=INK))
    ax4.set_xlabel("Per-point displacement", fontsize=8)
    ax4.set_ylabel("Points", fontsize=8)
    ax4.set_title("(d) Displacement", fontsize=8.5)
    ax4.tick_params(labelsize=7.5)
    ax4.grid(alpha=0.2, lw=0.5)
    for side in ("top", "right"):
        ax4.spines[side].set_visible(False)

    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        out = os.path.join(FIGDIR, f"denoising_example.{ext}")
        fig.savefig(out, bbox_inches="tight")
        print("saved", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
