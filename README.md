# Jittor AI Challenge, Track 2 — Point-Cloud Denoising & Classification

A solo entry to the 6th Jittor AI Challenge, implemented in [Jittor](https://github.com/Jittor/jittor): a reproduction of **StraightPCF** (CVPR 2024) for point-cloud denoising, and a **PCT** classifier for the qualification round.

**Technical report: [`report/tech_report.pdf`](report/tech_report.pdf)** — the method, the results, and what did not work. ([LaTeX source](report/tech_report.tex) · [中文说明](README.zh-CN.md))

## Results

| Task | Model | Result |
|---|---|---|
| Point-cloud denoising | StraightPCF reproduction: coupled velocity module, ≈0.7M parameters | **67.44/100** by the competition formula on our held-out split (CD sub-score 51.9, P2S sub-score 82.9); national top 100 |
| Shape classification | PCT: 2 sample-and-group stages and 4 offset-attention blocks, ≈2.9M parameters | 12th in the warmup round; the round's pass mark was 80% test accuracy |

Pooled over the held-out samples, denoising cuts the Chamfer distance by 51% and the point-to-surface error by 67% against the noisy input. The paper reports higher accuracy on its own Gaussian-noise benchmarks, which are not comparable to this competition's Laplace noise; the report's post-mortem attributes the gap chiefly to a training-schedule deviation from the paper.

## Repository layout

```
├── denoise/                 # point-cloud denoising
│   ├── run.py               #   train / predict / debug entry point
│   ├── self_eval.py         #   held-out evaluation with CD + P2S
│   ├── evaluate.py          #   official-style evaluator
│   ├── profile.py           #   data-vs-compute profiling, batch-size probing
│   ├── vis_denoising.py     #   per-sample denoising diagnostics
│   ├── configs/             #   YAML configs: task / data / model / system / transform
│   └── src/
│       ├── data/            #   mesh sampling, normalization, noise, patch construction
│       ├── model/           #   EdgeConv encoder, velocity modules, distance module
│       └── system/          #   trainer, and the writer that saves results
├── warmup/                  # ModelNet40 classification (PCT)
│   ├── train.py             #   training and inference
│   ├── rf_pct.py            #   PCT building blocks: offset attention, sample-and-group
│   └── rf_ops.py            #   low-level ops: FPS, k-NN, ball query
├── report/                  # technical report
│   ├── tech_report.pdf
│   ├── tech_report.tex
│   └── make_figures.py      #   regenerates the report figures from the run logs
└── README.zh-CN.md          # Chinese version of this README
```

## Quick start

Environment:

```bash
conda create -n jittor python=3.9 -y
conda activate jittor
conda install -c conda-forge gcc=10 gxx=10 libgomp -y
pip install jittor numpy trimesh scipy omegaconf matplotlib
pip install point-cloud-utils   # optional, for exact point-to-surface evaluation
```

Denoising, run from `denoise/`:

```bash
cd denoise

# Train, single GPU
python run.py --task configs/task/train_vm.yaml

# Train with MPI data parallelism
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" mpirun -np 6 python run.py --task configs/task/train_vm.yaml

# Inference; set load_ckpt in the config first
python run.py --task configs/task/predict_vm.yaml

# Held-out self-evaluation
python self_eval.py --task configs/task/train_vm.yaml --split_ratio 0.1 --num_samples 10000
```

The paper's staged schedule has its own configs: `configs/task/train_vm_stage{1..4}.yaml`. A score-based variant in the style of ScoreDenoise is implemented in `src/model/score_vm.py`, with `train_score.yaml` and `predict_score.yaml`; it was never trained to a comparable state.

Classification, run from `warmup/`:

```bash
cd warmup
python train.py --data_dir ./data --epochs 300 --batch_size 32
```

## Data

Training meshes and pre-noised test clouds are provided by the competition organizers and are not included here. The code expects them under `denoise/dataset_train/` and `denoise/dataset_test_noisy/`, with split lists in `denoise/datalist/`; `denoise/README.md` documents the layout.

## Documentation

| Document | Language | Contents |
|---|---|---|
| [`report/tech_report.pdf`](report/tech_report.pdf) | English | Method, Jittor engineering, results, post-mortem |
| [`denoise/README.md`](denoise/README.md) | Chinese | Denoising: usage, configs, packaging, FAQ |
| [`warmup/README.md`](warmup/README.md) | Chinese | Classification: architecture, training strategy, iteration history |
| [`denoise/ANALYSIS.md`](denoise/ANALYSIS.md) | Chinese | Working notes on the score regression and the staged-reproduction plan |

## Status

The competition ended in 2026; the code is archived as it was.

## License

MIT. See [LICENSE](LICENSE).

## References

- StraightPCF: *Straight Point Cloud Filtering*, CVPR 2024.
- ScoreDenoise: *Score-Based Point Cloud Denoising*, ICCV 2021 — the score-based variant in `src/model/score_vm.py`.
- PCT: *Point Cloud Transformer*, Computational Visual Media 2021 — the classifier used in the qualification round.
