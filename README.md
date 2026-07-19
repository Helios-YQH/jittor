# 计图挑战赛赛道二

本仓库包含计图（Jittor）挑战赛赛道二的全部代码：

| 赛题 | 目录 | 任务 | 模型 | 框架 |
|------|------|------|------|------|
| 热身赛 | [`warmup/`](warmup/) | ModelNet40 点云分类 | PCT (Point Cloud Transformer) | Jittor |
| 正式赛 | [`denoise/`](denoise/) | ShapeNet 点云去噪 | StraightPCF (Coupled Velocity Module) | Jittor |

## 目录结构

```
├── warmup/                 # 热身赛：点云分类
│   ├── train.py            #   训练 + 推理主脚本
│   ├── rf_pct.py           #   PCT 基础模块
│   ├── rf_ops.py           #   CUDA 底层算子
│   ├── show.py             #   数据增强可视化
│   └── README.md
├── denoise/                # 正式赛：点云去噪
│   ├── run.py              #   主入口
│   ├── evaluate.py         #   官方评测脚本
│   ├── self_eval.py        #   自测评脚本
│   ├── profile.py          #   性能分析
│   ├── vis_denoising.py    #   可视化诊断
│   ├── configs/            #   YAML 配置文件
│   ├── src/                #   源代码 (data/model/system)
│   ├── docs/               #   开发文档
│   └── README.md
└── papers/                 # 参考论文
    ├── StraightPCF.md      #   StraightPCF 论文精读
    ├── Straight_Point_Cloud_Filtering.pdf
    ├── P2P-Bridge_DiffusionBridgesfor3DPointCloudDenoising.pdf
    ├── DL_for_denoising.pdf
    ├── pct.pdf
    ├── PointNet.pdf
    └── PointNeXt.pdf
```

## 环境安装

```bash
conda create -n jittor python=3.9 -y
conda activate jittor
conda install -c conda-forge gcc=10 gxx=10 -y
conda install -c conda-forge libgomp -y
pip install jittor numpy trimesh scipy omegaconf matplotlib
```

详见各子目录的 README。
