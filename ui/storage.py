"""Файловый журнал и запись таблицы кластеризации."""

import csv
import logging
from pathlib import Path

import numpy as np

from fuzzy_numbers import FuzzyNumber, exponential_membership


class Journal:
    def __init__(self, root: Path):
        try:
            self.handler = self._open_handler(root / 'logs')
        except OSError:
            self.handler = self._open_handler(Path.home() / '.fuzzy_exponential_pyqt6')
        self.path = Path(self.handler.baseFilename)
        self.logger = logging.Logger('fuzzy_exponential_pyqt6', logging.INFO)
        self.logger.addHandler(self.handler)

    @staticmethod
    def _open_handler(directory: Path) -> logging.FileHandler:
        directory.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(directory / 'app.log', encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s', '%Y-%m-%d %H:%M:%S'))
        return handler

    def info(self, message: str, *args):
        self.logger.info(message, *args)

    def recent(self) -> str:
        try:
            return '\n'.join(self.path.read_text(encoding='utf-8').splitlines()[-700:])
        except OSError:
            return ''

    def copy_to(self, destination: Path):
        self.handler.flush()
        destination.write_text(self.path.read_text(encoding='utf-8'), encoding='utf-8')

    def close(self):
        self.logger.removeHandler(self.handler)
        self.handler.close()


def write_samples_csv(path: str, samples: list[FuzzyNumber], lam: float, alpha: float,
                      mask: np.ndarray, labels: np.ndarray | None, filtered_labels: dict):
    membership = exponential_membership([number.center for number in samples], lam)
    with open(path, 'w', newline='', encoding='utf-8-sig') as fh:
        writer = csv.writer(fh, delimiter=';')
        writer.writerow(['id', 'left', 'right', 'lambda', 'alpha', 'membership',
                         'selected', 'cluster', 'filtered_cluster'])
        for idx, number in enumerate(samples):
            writer.writerow([idx + 1, f'{number.left:.8g}', f'{number.right:.8g}', lam, alpha,
                             f'{membership[idx]:.8g}', int(mask[idx]),
                             '' if labels is None else int(labels[idx]), filtered_labels.get(idx, '')])
