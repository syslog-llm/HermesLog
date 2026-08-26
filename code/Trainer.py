import logging
import torch
import transformers
from torch.cuda.amp import autocast
from torch.nn import CrossEntropyLoss

class CustomTrainer_noKL_noSelfLoss(transformers.Trainer):
    def __init__(self, log_save_path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_save_path = log_save_path
        self.loss_stats = {
            'total': 0.0,
            'steps': 0
        }

    def log(self, logs, start_time=None):
        epoch = self.state.epoch if hasattr(self.state, "epoch") else None
        if "loss" in logs:
            self.loss_stats['total'] += logs["loss"]
            self.loss_stats['steps'] += 1

        avg_loss = self.loss_stats['total'] / self.loss_stats['steps'] if self.loss_stats['steps'] > 0 else 0.0

        logs.update({
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "total_loss": avg_loss,
            "epoch": epoch
        })

        self.loss_stats = {k: 0.0 for k in self.loss_stats}

        logging.info(f"Epoch {epoch} - total_loss:{logs['total_loss']:.4f} - learning_rate:{logs['learning_rate']:.6f}")
        with open(f"{self.log_save_path}/training_log.txt", "a") as f:
            f.write(f"Epoch {epoch} - total_loss:{logs['total_loss']:.4f} - learning_rate:{logs['learning_rate']:.6f}\n\n")
        
        print(logs)
        if epoch == self.state.epoch:
            super().log(logs, start_time)
