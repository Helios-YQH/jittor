"""Generate the figures for report/tech_report.tex.

Panel (a): Stage 1/2 pretraining loss (Eq. 7 velocity loss), July 2026 runs;
           one loss value per epoch, taken from each run's training.log.
Panel (b): the full coupled training run of May 2026, parsed from
           data/may_run_training.log (the original run log).

Outputs figures/*.pdf (used by LaTeX) and figures/*.png (300 dpi).
"""

import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
LOG_MAY = os.path.join(HERE, "data", "may_run_training.log")

# Stage 1/2 runs (July 2026), one optimizer epoch per loss value
STAGE1 = [3.7800, 0.4961, 0.3144, 0.1887, 0.1420, 0.1255, 0.1178, 0.1151,
          0.1140, 0.1137, 0.1135, 0.1129, 0.1040, 0.1023, 0.1012]
STAGE2 = [3.7794, 0.4963, 0.3141, 0.1883, 0.1433, 0.1261, 0.1183, 0.1153,
          0.1142, 0.1140, 0.1137, 0.1130, 0.1039, 0.1022, 0.1013]

OKABE_ITO = {"blue": "#0072B2", "vermillion": "#D55E00",
             "green": "#009E73", "grey": "#666666"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9.5,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})


def parse_may_log(path):
    train, val = [], []
    pat = re.compile(r"Epoch\s+(\d+)\s+\|\s+Loss:\s+([\d.]+)\s+\|\s+Val:\s+([\d.]+)")
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = pat.search(line)
            if m:
                train.append(float(m.group(2)))
                val.append(float(m.group(3)))
    return np.array(train), np.array(val)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))

    # ---- (a) Stage 1/2 pretraining (Eq. 7 velocity loss) ----
    ep = np.arange(len(STAGE1))
    ax1.semilogy(ep, STAGE1, "-o", color=OKABE_ITO["blue"], lw=1.4, ms=3.5,
                 label="VM1 (Stage 1)")
    ax1.semilogy(ep, STAGE2, "--s", color=OKABE_ITO["vermillion"], lw=1.4, ms=3.5,
                 mfc="none", label="VM2 (Stage 2)")
    ax1.axvline(10.5, color=OKABE_ITO["grey"], lw=0.8, ls=":", zorder=0)
    ax1.annotate("cosine warm restart", xy=(10.5, 0.5), xytext=(4.2, 2.0),
                 fontsize=8, color=OKABE_ITO["grey"],
                 arrowprops=dict(arrowstyle="->", lw=0.7, color=OKABE_ITO["grey"]))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Velocity loss  (Eq. 7)")
    ax1.set_title("(a) Stage 1/2 pretraining")
    ax1.set_xticks([0, 5, 10, 14])
    ax1.grid(True, which="both", alpha=0.25, lw=0.5)
    ax1.legend(loc="upper right", frameon=False)

    # ---- (b) full coupled training run, May 2026 ----
    train, val = parse_may_log(LOG_MAY)
    ep2 = np.arange(len(train))
    ax2.plot(ep2, train, "-", color=OKABE_ITO["blue"], lw=1.2, label="Train")
    ax2.plot(ep2 + 0.5, val, "--", color=OKABE_ITO["green"], lw=1.2, alpha=0.85,
             label="Validation")
    best = int(np.argmin(val))
    ax2.plot(best + 0.5, val[best], "v", color=OKABE_ITO["vermillion"], ms=5,
             label=f"Best validation ({val[best]:.4f})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Coupled training loss")
    ax2.set_title("(b) Full run (coupled objective)")
    ax2.set_xticks([0, 20, 40, 60, 80, 100])
    ax2.grid(True, alpha=0.25, lw=0.5)
    ax2.legend(loc="upper right", frameon=False)

    fig.tight_layout(pad=0.6)
    for ext in ("pdf", "png"):
        out = os.path.join(FIGDIR, f"training_curves.{ext}")
        fig.savefig(out, bbox_inches="tight")
        print("saved", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
