#!/usr/bin/env python
"""测速脚本：诊断单步训练瓶颈在数据加载还是 GPU 计算。

Usage:
    python profile.py                   # 自动从 16 向下探测可用 batch_size
    python profile.py --batch_size 8    # 手动指定
"""

import time
import argparse
import jittor as jt
jt.flags.use_cuda = 1

from omegaconf import OmegaConf
from src.data.dataset import DatasetConfig, PCDatasetModule
from src.data.transform import Transform
from src.model.parse import get_model


def load_cfg(path):
    if not path.endswith(('.yaml', '.yml')):
        path += '.yaml'
    return OmegaConf.to_container(OmegaConf.load(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    task = load_cfg('configs/task/train_vm.yaml')
    data_cfg = load_cfg('configs/data/train')
    transform_cfg = load_cfg('configs/transform/vm')
    model_cfg = load_cfg('configs/model/vm')

    model = get_model(model_config=model_cfg, transform_config=transform_cfg)
    model.train()

    # 自动探测可用 batch_size
    bs = args.batch_size
    if bs == 0:
        for trial_bs in [16, 12, 8, 6, 4]:
            bs = trial_bs
            print(f"尝试 batch_size={bs} ...")
            train_cfg = DatasetConfig.parse(**data_cfg['train_dataset'])
            train_cfg.batch_size = bs
            ds_module = PCDatasetModule(
                process_fn=model._process_fn,
                train_dataset_config=train_cfg,
                train_transform=model.get_train_transform(),
            )
            dl = ds_module.train_dataloader()
            assert dl is not None, "train_dataloader is None"
            try:
                for batch in dl:
                    _ = model.training_step(batch)
                    jt.sync_all()
                    break
                print(f"  batch_size={bs} OK")
                break  # succeeded
            except Exception as e:
                jt.clean_graph()
                jt.gc()
                print(f"  batch_size={bs} OOM: {e}")
                continue
    else:
        train_cfg = DatasetConfig.parse(**data_cfg['train_dataset'])
        train_cfg.batch_size = bs
        ds_module = PCDatasetModule(
            process_fn=model._process_fn,
            train_dataset_config=train_cfg,
            train_transform=model.get_train_transform(),
        )
        dl = ds_module.train_dataloader()
        assert dl is not None, "train_dataloader is None"

    print(f"\n使用 batch_size={bs}, num_workers={dl.num_workers}")

    # 释放预热阶段的中间变量
    jt.clean_graph()
    jt.gc()

    # 正式测速
    n_steps = args.steps
    load_times = []
    gpu_times = []
    total_times = []

    t0 = time.time()
    step = 0
    for batch in dl:
        t_load = time.time() - t0

        t_gpu_start = time.time()
        loss_dict = model.training_step(batch)
        jt.sync_all()
        loss_val = float(loss_dict["loss"].data)
        t_gpu = time.time() - t_gpu_start

        load_times.append(t_load)
        gpu_times.append(t_gpu)
        total_times.append(t_load + t_gpu)

        step += 1
        if step >= n_steps:
            break
        t0 = time.time()

    # 结果
    print(f"\n{'='*55}")
    print(f"  步数: {n_steps}")
    print(f"  batch_size: {bs}")
    print(f"  loss (最后一步): {loss_val:.6f}")
    print(f"{'='*55}")
    print(f"  {'指标':<20} {'均值':>10} {'最小':>10} {'最大':>10}")
    print(f"  {'数据加载 (s/step)':<20} {sum(load_times)/n_steps:>10.3f} {min(load_times):>10.3f} {max(load_times):>10.3f}")
    print(f"  {'GPU 计算 (s/step)':<20} {sum(gpu_times)/n_steps:>10.3f} {min(gpu_times):>10.3f} {max(gpu_times):>10.3f}")
    print(f"  {'总计 (s/step)':<20} {sum(total_times)/n_steps:>10.3f} {min(total_times):>10.3f} {max(total_times):>10.3f}")
    print(f"{'='*55}")

    load_pct = sum(load_times) / sum(total_times) * 100
    gpu_pct = sum(gpu_times) / sum(total_times) * 100
    print(f"\n瓶颈分析:")
    print(f"  数据加载占比: {load_pct:.1f}%")
    print(f"  GPU 计算占比: {gpu_pct:.1f}%")

    if load_pct > 50:
        print(f"\n  ⚠️ 瓶颈在 CPU 数据预处理。分布式训练收益受限。")
        print(f"     建议：增加 num_workers / 预处理缓存 / 减少采样点数")
    elif load_pct > 30:
        print(f"\n  ⚡ CPU/GPU 较均衡。分布式训练有效，预估 6 卡提速 3-4×")
    else:
        print(f"\n  ✅ 瓶颈在 GPU 计算。分布式训练收益最大。")
        print(f"     预估 6 卡提速 4-5×")

    print(f"\n  (单卡批大小={bs}，6 卡有效批={bs*6})")


if __name__ == '__main__':
    main()
