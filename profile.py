#!/usr/bin/env python
"""测速脚本：诊断单步训练瓶颈在数据加载还是 GPU 计算。

Usage:
    python profile.py
"""

import time
import jittor as jt
jt.flags.use_cuda = 1

from omegaconf import OmegaConf
from src.data.dataset import DatasetConfig, PCDatasetModule
from src.data.transform import Transform
from src.model.parse import get_model
from src.system.parse import get_system


def load_cfg(path):
    return OmegaConf.to_container(OmegaConf.load(path.removesuffix('.yaml') + '.yaml'))


def main():
    # 加载配置（使用正式训练配置确保测速结果有代表性）
    task = load_cfg('configs/task/train_vm.yaml')
    data_cfg = load_cfg('configs/data/train')
    transform_cfg = load_cfg('configs/transform/vm')
    model_cfg = load_cfg('configs/model/vm')

    # 构建 model
    model = get_model(model_config=model_cfg, transform_config=transform_cfg)

    # 构建 dataset（只用训练部分）
    train_cfg = DatasetConfig.parse(**data_cfg['train_dataset'])
    train_transform = model.get_train_transform()

    ds_module = PCDatasetModule(
        process_fn=model._process_fn,
        train_dataset_config=train_cfg,
        train_transform=train_transform,
    )

    dl = ds_module.train_dataloader()
    assert dl is not None

    print(f"batch_size={dl.batch_size}, num_workers={dl.num_workers}")

    # 预热一轮
    model.train()
    for batch in dl:
        loss = model.training_step(batch)
        break

    # 开始测速
    n_steps = 20
    load_times = []
    gpu_times = []
    total_times = []

    t0 = time.time()
    step = 0
    for batch in dl:
        t_load = time.time() - t0
        jt.sync_all()  # 清空异步队列

        t_gpu_start = time.time()
        loss = model.training_step(batch)
        loss_val = loss.item()  # 强制同步
        jt.sync_all()
        t_gpu = time.time() - t_gpu_start

        load_times.append(t_load)
        gpu_times.append(t_gpu)
        total_times.append(t_load + t_gpu)

        step += 1
        if step >= n_steps:
            break
        t0 = time.time()

    # 打印结果
    print(f"\n{'='*55}")
    print(f"  步数: {n_steps}")
    print(f"  batch_size: {dl.batch_size}")
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

    # 显存报告
    mem = jt.memory_info()
    print(f"\n显存: 已分配 {mem[1]/1024:.0f} MB / 总 {mem[0]/1024:.0f} MB ({mem[1]/mem[0]*100:.0f}%)")


if __name__ == '__main__':
    main()
