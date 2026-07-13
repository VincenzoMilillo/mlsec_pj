import csv
from pathlib import Path


class CSVLogger:
    """Minimal CSV logger compatible with pytorch-lightning's logger API.

    This does not subclass LightningLoggerBase to avoid hard dependency on the
    internal import path. It implements the methods Lightning expects:
    - log_metrics(metrics, step)
    - log_hyperparams(hparams)
    - properties: name, version, experiment
    """

    def __init__(self, train_csv_path, val_csv_path, train_header, val_header):
        self.train_csv_path = train_csv_path
        self.val_csv_path = val_csv_path
        # minimal attribute Lightning expects
        self.save_dir = str(Path.cwd())

        with open(self.train_csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(train_header)

        with open(self.val_csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(val_header)

    def log_metrics(self, metrics, step=None):
        # Support both Lightning's metric naming and the original project's keys
        if 'train_loss_epoch' in metrics or 'train_loss' in metrics:
            epoch = metrics.get('epoch', None)
            train_loss = metrics.get('train_loss_epoch', metrics.get('train_loss'))
            train_acc = metrics.get('train_acc_epoch', metrics.get('train_acc'))
            fields = [epoch, train_loss, train_acc]
            filename = self.train_csv_path
        elif 'val_loss' in metrics or 'val_loss_epoch' in metrics:
            epoch = metrics.get('epoch', None)
            val_loss = metrics.get('val_loss', metrics.get('val_loss_epoch'))
            val_acc = metrics.get('val_acc', metrics.get('val_acc_epoch'))
            fields = [epoch, val_loss, val_acc]
            filename = self.val_csv_path
        else:
            return

        with open(filename, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)

    @property
    def experiment(self):
        return None

    @property
    def name(self):
        return 'csvlogger'

    def log_hyperparams(self, hparams):
        # no-op for CSV logger
        return

    @property
    def version(self):
        return '0'

    def finalize(self, status: str):
        # Lightning calls finalize on completion or failure; nothing to do here.
        return

    def log_graph(self, *args, **kwargs):
        # Lightning may call this to log the model graph; no-op for CSV logger
        return

    @property
    def log_dir(self):
        # Some Lightning utilities access log_dir; provide alias to save_dir
        return self.save_dir

    def save(self):
        # no-op: Lightning may call save() on loggers
        return

    def after_save_checkpoint(self, checkpoint_callback):
        # no-op: Lightning calls this after checkpoint saves.
        return
