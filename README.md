# Jittor AI Challenge, Track 2 — Point-Cloud Denoising & Classification

A solo entry to the **6th Jittor AI Challenge** (Track 2), implemented in [Jittor](https://github.com/Jittor/jittor):

- **Main track — point-cloud denoising**: a reproduction of **StraightPCF** (CVPR 2024) that predicts per-point displacements returning noisy ShapeNet point clouds to their surfaces.
- **Qualification round — shape classification**: a **PCT** (Point Cloud Transformer) model for ModelNet40.

**Technical report: [`report/tech_report.pdf`](report/tech_report.pdf)** — the method as implemented, the Jittor engineering it required, measured results, and an honest post-mortem of what did not work. ([LaTeX source](report/tech_report.tex) · [中文说明](README.zh-CN.md))

## Results at a glance

| Track | Model | Result |
|---|---|---|
| Denoising (main) | StraightPCF reproduction — coupled velocity module, ≈0.7M params | **67.44/100** best competition score (CD sub-score 51.9, P2S sub-score 82.9); pooled CD −51% and P2S −67% vs. the noisy input; national top-100 placement |
| Classification (qualification) | PCT — 2 sample-and-group stages + 4 offset-attention blocks, ≈2.9M params | Cleared the qualification round (≥80% test-accuracy threshold) |

Numbers are measured on a held-out split of the training meshes with the competition metric (Chamfer distance and point-to-surface distance, per-sample scoring). The reproduction does **not** match the accuracy reported in the StraightPCF paper; the report's post-mortem section gives our best account of why.

## Repository layout

```
├── denoise/                 # main track: point-cloud denoising
│   ├── run.py               #   train / predict / debug entry point
│   ├── self_eval.py         #   held-out evaluation (CD + P2S, competition scoring)
│   ├── evaluate.py          #   official-style evaluator
│   ├── profile.py           #   data-vs-compute profiling, batch-size probing
│   ├── vis_denoising.py     #   per-sample denoising diagnostics
│   ├── configs/             #   YAML configs: task / data / model / system / transform
│   └── src/                 #   data pipeline, models, training loop
│       ├── data/            #     mesh sampling, normalization, noise, patch construction
│       ├── model/           #     EdgeConv encoder, velocity modules, distance module
│       └── system/          #     trainer (MPI, checkpointing, scheduling), result writer
├── warmup/                  # qualification round: ModelNet40 classification (PCT)
│   ├── train.py             #   training + inference
│   ├── rf_pct.py            #   PCT building blocks (offset attention, sample-and-group)
│   └── rf_ops.py            #   low-level ops (FPS, k-NN, ball query)
├── report/                  # technical report: LaTeX source, PDF, figures, run logs
│   ├── tech_report.pdf
│   ├── tech_report.tex
│   └── make_figures.py      #   regenerates the report's figures from the run logs
└── README.zh-CN.md          # Chinese version of this README
```

## Quick start

Environment (from `denoise/README.md`):

```bash
conda create -n jittor python=3.9 -y
conda activate jittor
conda install -c conda-forge gcc=10 gxx=10 libgomp -y
pip install jittor numpy trimesh scipy omegaconf matplotlib
pip install point-cloud-utils   # optional, for exact point-to-surface evaluation
```

Denoising (paths are relative to `denoise/`):

```bash
cd denoise

# Train (single GPU; the competition configuration)
python run.py --task configs/task/train_vm.yaml

# Train with MPI data parallelism
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" mpirun -np 6 python run.py --task configs/task/train_vm.yaml

# Inference (set load_ckpt in the config first)
python run.py --task configs/task/predict_vm.yaml

# Held-out self-evaluation (CD + P2S, competition scoring)
python self_eval.py --task configs/task/train_vm.yaml --split_ratio 0.1 --num_samples 10000
```

The paper's staged schedule has its own configs: `configs/task/train_vm_stage{1..4}.yaml`. A score-based variant (ScoreDenoise-style) lives in `src/model/score_vm.py` with configs `train_score.yaml` / `predict_score.yaml`; it was never trained to a competitive state.

Classification (warmup):

```bash
cd warmup
python train.py --data_dir ./data --epochs 300 --batch_size 32
```

## Data

The datasets — ShapeNet meshes for training and the pre-noised test clouds — are provided by the competition organizers and are **not** redistributed in this repository. The code expects them under `denoise/dataset_train/` and `denoise/dataset_test_noisy/`, with split lists in `denoise/datalist/`. See `denoise/README.md` for the expected layout.

## Documentation

| Document | Language | Contents |
|---|---|---|
| [`report/tech_report.pdf`](report/tech_report.pdf) | English | Method, Jittor engineering, results, post-mortem |
| [`denoise/README.md`](denoise/README.md) | Chinese | Denoising: full usage, configs, packaging, FAQ |
| [`warmup/README.md`](warmup/README.md) | Chinese | Classification: architecture, training strategy, iteration history |
| [`denoise/ANALYSIS.md`](denoise/ANALYSIS.md) | Chinese | Working notes on the score regression and the staged-reproduction plan |

## Status

The competition has ended (2026). The repository is kept as a record: the code is archived as it was, and `report/tech_report.pdf` documents both the parts that worked and the parts that did not.

## References

- StraightPCF: *Straight Point Cloud Filtering*, CVPR 2024 — the method reproduced here.
- ScoreDenoise: *Score-Based Point Cloud Denoising*, ICCV 2021 — the score-based variant in `src/model/score_vm.py`.
- PCT: *Point Cloud Transformer*, Computational Visual Media 2021 — the warmup-round classifier.
