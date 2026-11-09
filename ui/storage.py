"""Файловый журнал и запись таблицы кластеризации."""

import csv
import logging
from pathlib import Path

import numpy as np

from clustering import Sample


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


def write_samples_csv(path: str, samples: list[Sample], labels: np.ndarray | None):
    with open(path, 'w', newline='', encoding='utf-8-sig') as fh:
        writer = csv.writer(fh, delimiter=';')
        writer.writerow(['id', 'a', 'lambda', 'alpha', 'left', 'right', 'width', 'cluster'])
        for idx, sample in enumerate(samples):
            writer.writerow([idx + 1, f'{sample.a:.8f}', f'{sample.lam:.8f}',
                             f'{sample.number.alpha:.5f}',
                             f'{sample.number.left[0]:.8f}', f'{sample.number.right[0]:.8f}',
                             f'{sample.number.right[0] - sample.number.left[0]:.8f}',
                             '' if labels is None else int(labels[idx])])
