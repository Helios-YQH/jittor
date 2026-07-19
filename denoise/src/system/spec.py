from collections import defaultdict
from datetime import datetime, timezone, timedelta
from jittor import optim
from typing import Dict, List, Optional
from tqdm import tqdm

import jittor as jt
import os
import time

from ..data.asset import Asset
from ..data.dataset import PCDatasetModule
from ..model.spec import ModelSpec

_CST = timezone(timedelta(hours=8))


def _now_cst():
    return datetime.now(_CST)


def _get_item(x):
    if isinstance(x, jt.Var):
        return x.item()
    return x


def _get_mpi_rank():
    """Get MPI rank via env var (OpenMPI standard). Returns 0 if not in MPI mode."""
    if jt.mpi is None:
        return 0
    try:
        return int(os.environ.get('OMPI_COMM_WORLD_RANK', 0))
    except (ValueError, TypeError):
        return 0


def _is_main_process():
    """True on rank 0 or single-GPU."""
    return _get_mpi_rank() == 0

def get_optimizer(optimizer_config, model):
    __target__ = optimizer_config.pop('__target__')
    MAPPING = {
        'sgd': optim.SGD,
        'adam': optim.Adam,
    }
    if __target__ not in MAPPING:
        raise ValueError(f"unsupported optimizer: {__target__}")
    OptimizerClass = MAPPING[__target__]
    optimizer = OptimizerClass(model.parameters(), **optimizer_config)
    return optimizer

class DummyWriter():
    
    def __init__(self):
        pass
    
    def write(self, batch, prediction: List[Dict], dataset_module: Optional[PCDatasetModule]=None):
        pass

class DummySystem():
    
    def __init__(
        self,
        dataset_module: PCDatasetModule,
        model: ModelSpec,
        loss_config=None,
        optimizer_config=None,
        trainer_config=None,
        writer: Optional[DummyWriter]=None,
        
        ckpt_save_dir: str="experiments",
        ckpt_save_name: str="checkpoint",
    ):
        self.dataset_module = dataset_module
        self.model = model
        self.loss_config = loss_config
        self.ckpt_save_dir = ckpt_save_dir
        self.ckpt_save_name = ckpt_save_name
        self.writer = writer
        if trainer_config is None:
            trainer_config = {}
        self.epochs = trainer_config.get('epochs', 1)
        
        if optimizer_config is not None and model is not None:
            self.optimizer = get_optimizer(optimizer_config, model)
        else:
            self.optimizer = None
        # scheduler config (optional) - read from trainer_config
        self.scheduler_config = trainer_config.get('scheduler') if isinstance(trainer_config, dict) else None
        self.phases = trainer_config.get('phases', None) if isinstance(trainer_config, dict) else None
        if self.scheduler_config is not None:
            self._base_lr = float(optimizer_config.get('lr', 1e-4)) if optimizer_config is not None else 1e-4
            self._scheduler_state = {'last_restart': 0, 'T_cur': 0}
        self.best_val = None
        self.patience = trainer_config.get('patience', 0) if isinstance(trainer_config, dict) else 0
        self._epochs_no_improve = 0
        self._validation_loss = defaultdict(list)
        self._train_loss_history = []
        self._val_loss_history = []
        # open log file (per-run subdirectory)
        os.makedirs(self.ckpt_save_dir, exist_ok=True)
        run_name = trainer_config.get('run_name', '') if isinstance(trainer_config, dict) else ''
        if not run_name:
            run_name = f"{self.ckpt_save_name}_{_now_cst().strftime('%Y%m%d_%H%M%S')}"
        self.run_name = run_name
        self.run_dir = os.path.join(self.ckpt_save_dir, self.run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.log_path = os.path.join(self.run_dir, 'training.log')
        try:
            with open(self.log_path, 'a') as f:
                f.write(f"\n{'='*60}\nRun: {self.run_name} started at {_now_cst()}\n{'='*60}\n")
        except Exception:
            pass
    
    def forward(self, batch, validate: bool=False): # return loss sum
        loss_dict = self.model.training_step(batch)
        assert isinstance(loss_dict, dict), "loss_dict must be a dict containing loss/metrics"
        assert self.loss_config is not None, "do not have loss_confing"
        loss_sum = 0.
        if validate:
            assets: List[Asset] = [a for a in batch['asset']]
            cls = assets[0].cls # guaranteed to be the same cls in dataloader
            for name in loss_dict:
                assert name in self.loss_config, f'unspecified loss {name}'
                self._validation_loss[f"val/{cls}_{name}"].append(_get_item(loss_dict[name]))
                loss_sum += self.loss_config[name] * loss_dict[name]
            self._validation_loss[f"val/{cls}_loss_sum"].append(_get_item(loss_sum))
            # TODO: log
            # self.log('val/loss_sum', loss_sum, prog_bar=True, logger=True, sync_dist=True, batch_size=len(assets))
        else:
            for name in loss_dict:
                assert name in self.loss_config, f"unspecified loss name: `{name}`"
                if self.loss_config[name] > 0:
                    loss_sum += self.loss_config[name] * loss_dict[name]
            loss_dict['loss_sum'] = loss_sum
            # TODO: log
            # # add train prefix to loss_dict
            # prefixed_loss_dict = {f"train/{k}": v for k, v in loss_dict.items()}
            # d = dict(sorted(prefixed_loss_dict.items()))
        if not isinstance(loss_sum, jt.Var):
            return jt.array(loss_sum)
        return loss_sum
    
    def on_train_epoch_start(self):
        pass
    
    def on_train_batch_start(self):
        pass
    
    def training_step(self, batch):
        return self.forward(batch, validate=False)
    
    def on_train_batch_end(self):
        pass
    
    def on_train_epoch_end(self):
        pass
    
    def on_validation_epoch_start(self):
        self._validation_loss = defaultdict(list)
    
    def on_validation_batch_start(self):
        pass
    
    def validation_step(self, batch):
        assert self.loss_config is not None, "do not have loss_confing"
        return self.forward(batch, validate=True)
    
    def on_validation_batch_end(self):
        pass
    
    def on_validation_epoch_end(self):
        # compute mean validation loss across recorded metrics and save best checkpoint
        try:
            vals = []
            for k, v in self._validation_loss.items():
                if isinstance(v, list) and len(v) > 0:
                    vals.extend(v)
            if len(vals) > 0:
                mean_val = sum(vals) / len(vals)
            else:
                mean_val = None
        except Exception:
            mean_val = None
        # all-reduce validation loss across distributed workers
        if mean_val is not None and jt.mpi:
            try:
                mean_val = jt.array(mean_val).mpi_all_reduce("mean")
                if isinstance(mean_val, jt.Var):
                    mean_val = mean_val.item()
            except Exception:
                pass
        # save best model (rank 0 only) and track early stopping
        self._last_val_loss = mean_val
        try:
            if mean_val is not None:
                if self.best_val is None or mean_val < self.best_val:
                    self.best_val = mean_val
                    self._epochs_no_improve = 0
                    best_path = os.path.join(self.run_dir, f'{self.ckpt_save_name}_best.pkl')
                    if _is_main_process():
                        self.model.save(best_path)
                    try:
                        with open(self.log_path, 'a') as f:
                            f.write(f"New best validation {mean_val}\n")
                    except Exception:
                        pass
                else:
                    self._epochs_no_improve += 1
        except Exception:
            pass
    
    def on_before_optimizer_step(self, optimizer):
        pass

    def _plot_curve(self):
        """Save training curve as PNG in run directory."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            epochs = range(len(self._train_loss_history))
            ax.plot(epochs, self._train_loss_history, 'b-', label='Train Loss', linewidth=1.5)
            if self._val_loss_history:
                ax.plot(epochs, self._val_loss_history, 'r-', label='Val Loss', linewidth=1.5)
                best_epoch = min(range(len(self._val_loss_history)), key=lambda i: self._val_loss_history[i])
                ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best Val ({self.best_val:.4f})')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(f'Training Curve - {self.run_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.run_dir, 'training_curve.png'), dpi=150)
            plt.close()
        except Exception:
            pass
    
    def on_predict_epoch_start(self):
        pass
    
    def on_predict_batch_start(self):
        pass
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        return self.model.predict_step(batch)
    
    def on_predict_batch_end(self):
        pass
    
    def on_predict_epoch_end(self):
        pass

    def _apply_phase(self, epoch):
        """Apply phase-based freeze/unfreeze at epoch boundaries."""
        if self.phases is None:
            return
        current_phase = None
        for phase in self.phases:
            if phase['start'] <= epoch < phase['end']:
                current_phase = phase
                break
        if current_phase is None:
            return
        model = self.model
        if current_phase.get('freeze_backbone', False):
            if hasattr(model, 'freeze_backbone'):
                model.freeze_backbone()
        else:
            if hasattr(model, 'unfreeze_backbone'):
                model.unfreeze_backbone()
        if current_phase.get('freeze_distance', False):
            if hasattr(model, 'freeze_distance'):
                model.freeze_distance()
        else:
            if hasattr(model, 'unfreeze_distance'):
                model.unfreeze_distance()

    def _apply_scheduler(self, epoch):
        """CosineAnnealingWarmRestarts with warmup or StepLR."""
        if self.scheduler_config is None:
            return
        try:
            stype = self.scheduler_config.get('type')
            if stype == 'cosine_warm_restart':
                lr_min = float(self.scheduler_config.get('lr_min', 1e-6))
                T_0 = int(self.scheduler_config.get('T_0', 50))
                T_mult = int(self.scheduler_config.get('T_mult', 2))
                warmup_epochs = int(self.scheduler_config.get('warmup_epochs', 0))
                warmup_start_lr = float(self.scheduler_config.get('warmup_start_lr', self._base_lr * 0.1))

                # Determine current restart cycle period
                T_i = T_0
                cycle_start = 0
                restarts = 0
                temp_T = T_0
                while True:
                    next_cycle_start = cycle_start + temp_T
                    if epoch < next_cycle_start:
                        T_i = temp_T
                        break
                    cycle_start = next_cycle_start
                    temp_T *= T_mult
                    restarts += 1
                    if restarts > 100:
                        break

                T_cur = epoch - cycle_start

                # Warmup: linear from warmup_start_lr to base_lr
                if warmup_epochs > 0 and epoch < warmup_epochs:
                    progress = epoch / max(warmup_epochs, 1)
                    new_lr = warmup_start_lr + (self._base_lr - warmup_start_lr) * progress
                else:
                    # Cosine annealing within current cycle
                    effective_epoch = T_cur - warmup_epochs if cycle_start == 0 and T_cur < warmup_epochs else T_cur
                    effective_T = T_i - warmup_epochs if cycle_start == 0 else T_i
                    if effective_T <= 0:
                        effective_T = T_i
                    progress = min(effective_epoch / max(effective_T, 1), 1.0)
                    new_lr = lr_min + 0.5 * (self._base_lr - lr_min) * (1.0 + np.cos(np.pi * progress))

                new_lr = max(new_lr, lr_min)
                for g in self.optimizer.param_groups:
                    g['lr'] = new_lr
                return new_lr

            elif stype == 'step':
                step = int(self.scheduler_config.get('step_size', 30))
                gamma = float(self.scheduler_config.get('gamma', 0.1))
                if (epoch + 1) % step == 0:
                    for g in self.optimizer.param_groups:
                        g['lr'] = g.get('lr', self._base_lr) * gamma
        except Exception:
            pass
    
    def train(self):
        assert self.optimizer is not None, "optimizer is None, cannot train"
        self.model.set_predict(False)
        # broadcast parameters in distributed mode
        if jt.mpi:
            self.model.mpi_param_broadcast(root=0)
        disable_pbar = not _is_main_process()

        # Create dataloaders ONCE before training — avoid forking workers
        # after CUDA context is active (CUDA+fork = undefined behavior).
        train_dataloader = self.dataset_module.train_dataloader()
        assert train_dataloader is not None, "train_dataloader is None"
        validate_dataloader = self.dataset_module.validate_dataloader()

        for epoch in range(self.epochs):
            # Apply phase-based freeze/unfreeze BEFORE training this epoch
            self._apply_phase(epoch)

            self.model.train()
            self.on_train_epoch_start()
            pbar = tqdm(train_dataloader, total=len(train_dataloader)//train_dataloader.batch_size, disable=disable_pbar) # type: ignore
            epoch_losses = []
            t_ep_start = time.time()
            for batch in pbar:
                self.on_train_batch_start()
                loss = self.training_step(batch)
                self.optimizer.zero_grad()
                self.optimizer.backward(loss)
                loss_val = _get_item(loss)
                epoch_losses.append(loss_val)
                pbar.set_description(f"Epoch {epoch}, Loss: {loss_val:.4f}")
                self.on_before_optimizer_step(self.optimizer)
                self.optimizer.step()
                jt.sync_all()  # execute pending ops and release GPU memory
                jt.gc()       # force Jittor garbage collection
                self.on_train_batch_end()
            self.on_train_epoch_end()
            
            self.model.eval()
            if validate_dataloader is not None:
                self.on_validation_epoch_start()
                if isinstance(validate_dataloader, dict):
                    for name, dataloader in validate_dataloader.items():
                        pbar = tqdm(dataloader, total=len(dataloader)//dataloader.batch_size, disable=disable_pbar)
                        for batch in pbar:
                            self.on_validation_batch_start()
                            loss = self.validation_step(batch)
                            pbar.set_description(f"Epoch {epoch}, Validate {name}, Loss: {_get_item(loss)}")
                            self.on_validation_batch_end()
                else:
                    pbar = tqdm(validate_dataloader, total=len(validate_dataloader)//validate_dataloader.batch_size, disable=disable_pbar)
                    for batch in pbar:
                        self.on_validation_batch_start()
                        loss = self.validation_step(batch)
                        pbar.set_description(f"Epoch {epoch}, Validate, Loss: {_get_item(loss)}")
                        self.on_validation_batch_end()
                self.on_validation_epoch_end()
                jt.sync_all()
                jt.gc()

            # epoch summary
            if _is_main_process():
                t_ep = time.time() - t_ep_start
                mean_loss = sum(epoch_losses) / len(epoch_losses)
                # Jittor 的 Adam 初始化时 param_groups 可能不含 'lr' 键，
                # 回退读取 optimizer.lr 属性，并统一转为 float 格式化
                lr = self.optimizer.param_groups[0].get('lr', None)
                if lr is None and hasattr(self.optimizer, 'lr'):
                    lr = getattr(self.optimizer, 'lr', None)
                if isinstance(lr, jt.Var):
                    lr = lr.item()
                if isinstance(lr, (int, float)):
                    lr_str = f"{lr:.2e}" if lr < 1e-4 else str(lr)
                else:
                    lr_str = '?'
                best_str = f"{self._last_val_loss:.4f}" if self._last_val_loss is not None else "N/A"
                flag = " ★" if self._epochs_no_improve == 0 and self.best_val is not None else ""
                print(f"Epoch {epoch:3d}/{self.epochs} | "
                      f"Loss: {mean_loss:.4f} | Val: {best_str}{flag} | "
                      f"LR: {lr_str} | {t_ep:.1f}s")
                try:
                    with open(self.log_path, 'a') as f:
                        f.write(f"Epoch {epoch:3d} | Loss: {mean_loss:.4f} | Val: {best_str}{flag} | LR: {lr_str} | {t_ep:.1f}s\n")
                except Exception:
                    pass
                self._train_loss_history.append(mean_loss)
                self._val_loss_history.append(self._last_val_loss if self._last_val_loss is not None else float('nan'))
                self._plot_curve()

            checkpoint_path = os.path.join(self.run_dir, f'{self.ckpt_save_name}_{epoch}.pkl')
            os.makedirs(self.run_dir, exist_ok=True)
            if _is_main_process():
                self.model.save(checkpoint_path)
            jt.sync_all()
            jt.gc()
            # update scheduler if configured
            if self.scheduler_config is not None:
                new_lr = self._apply_scheduler(epoch)
            # early stopping
            if self.patience > 0 and self._epochs_no_improve >= self.patience:
                if _is_main_process():
                    print(f"Early stopping: no improvement for {self.patience} epochs.")
                break
    
    def predict(self):
        # only iterate once
        self.model.set_predict(True)
        self.model.eval()
        self.on_predict_epoch_start()
        predict_dataloader = self.dataset_module.predict_dataloader()
        assert predict_dataloader is not None, "predict_dataloader is None"
        if not isinstance(predict_dataloader, dict):
            predict_dataloader = {"predict": predict_dataloader}
        for dataloader_name, dataloader in predict_dataloader.items():
            pbar = tqdm(dataloader, total=len(dataloader)//dataloader.batch_size) # type: ignore
            for batch_idx, batch in enumerate(pbar):
                self.on_predict_batch_start()
                output = self.predict_step(batch, batch_idx)
                if self.writer is not None:
                    self.writer.write(batch, output, dataset_module=self.dataset_module)
                pbar.set_description(f"Predicting {dataloader_name}, Batch {batch_idx}")