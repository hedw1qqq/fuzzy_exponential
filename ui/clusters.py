"""Генерация набора и фоновая кластеризация."""

from collections.abc import Callable

import numpy as np
from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QHeaderView, QMessageBox,
    QSplitter, QTableWidget, QTableWidgetItem, QWidget,
)

from clustering import METRICS, alpha_mask, generate_samples, k_medoids, pairwise_distances
from fuzzy_numbers import ALPHA, FuzzyNumber, exponential_membership
from .storage import Journal, write_samples_csv
from .widgets import CLUSTER_COLORS, Chart, button, card, field_layout, int_box, label, make_scroll, vertical


class ClusteringWorker(QObject):
    result = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, numbers, mask, k, metric, max_iter, seed):
        super().__init__()
        self.numbers = numbers
        self.mask = mask
        self.k = k
        self.metric = metric
        self.max_iter = max_iter
        self.seed = seed

    def _cluster(self, distances):
        labels, medoids = k_medoids(
            distances, min(self.k, len(distances)), self.max_iter, self.seed,
        )
        return labels, distances[np.arange(len(distances)), medoids[labels]]

    @pyqtSlot()
    def run(self):
        try:
            distances = pairwise_distances(self.numbers, self.metric)
            full = self._cluster(distances)
            filtered = self._cluster(distances[np.ix_(self.mask, self.mask)]) if self.mask.any() else None
            self.result.emit((full, filtered))
        except Exception as exc:
            self.error.emit(str(exc))


class ClustersPage(QWidget):
    busy_changed = pyqtSignal(bool)

    def __init__(self, write_log: Callable[[str], None], journal: Journal):
        super().__init__()
        self._write_log = write_log
        self.journal = journal
        self.alpha = ALPHA
        self.lam = 1.0
        self.samples: list[FuzzyNumber] = []
        self.full = self.filtered = None
        self.worker: ClusteringWorker | None = None
        self.thread: QThread | None = None
        self.setObjectName('page')
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(split)
        panel = QWidget()
        panel.setObjectName('page')
        stack = vertical(8, (0, 0, 4, 0))
        panel.setLayout(stack)
        generator = card(vertical(7, (14, 11, 14, 11)))
        g = generator.layout()
        g.addWidget(label('Генерация выборки', 'section'))
        self.n_samples = int_box(150, 100, 300)
        self.seed = int_box(42, 0, 2_147_483_647)
        sample_fields = QHBoxLayout()
        sample_fields.addLayout(field_layout('Интервалы', self.n_samples))
        sample_fields.addLayout(field_layout('Seed', self.seed))
        self.n_samples.setToolTip('От 100 до 300 интервалов')
        self.seed.setToolTip('Задаёт воспроизводимость выборки и начальных медоидов')
        g.addLayout(sample_fields)
        self.generate_btn = button('Сгенерировать')
        self.generate_btn.clicked.connect(self.generate_data)
        g.addWidget(self.generate_btn)
        stack.addWidget(generator)

        settings = card(vertical(7, (14, 11, 14, 11)))
        s = settings.layout()
        s.addWidget(label('Настройки k-medoids', 'section'))
        self.k_count = int_box(3, 1, 300)
        self.iterations = int_box(100, 1, 1000)
        self.metric = QComboBox()
        self.metric.addItems(('Хаусдорф', 'Разность ширин', 'Разность квадратов', 'Евклидова'))
        for index, metric_name in enumerate(METRICS):
            self.metric.setItemData(index, metric_name, Qt.ItemDataRole.ToolTipRole)
        self.metric.currentIndexChanged.connect(self._metric_changed)
        cluster_fields = QHBoxLayout()
        cluster_fields.addLayout(field_layout('Кластеры', self.k_count))
        cluster_fields.addLayout(field_layout('Макс. итераций', self.iterations))
        s.addLayout(cluster_fields)
        s.addLayout(field_layout('Метрика', self.metric))
        self.cluster_btn = button('Выполнить кластеризацию', primary=True)
        self.cluster_btn.clicked.connect(self.cluster_data)
        s.addWidget(self.cluster_btn)
        stack.addWidget(settings)
        stack.addStretch(1)
        scroll = make_scroll(panel)
        scroll.setMinimumWidth(260)
        split.addWidget(scroll)

        right = QWidget()
        right.setObjectName('page')
        right_layout = vertical(10, (0, 0, 0, 0))
        right.setLayout(right_layout)
        chart_frame = card(vertical(7, (14, 13, 14, 13)))
        self.cluster_chart = Chart(side_by_side=True)
        chart_frame.layout().addWidget(self.cluster_chart, 1)
        self.cluster_status = label('Сначала сгенерируйте набор.', 'hint', True)
        chart_frame.layout().addWidget(self.cluster_status)
        right_layout.addWidget(chart_frame, 5)
        table_frame = card(vertical(8, (14, 12, 14, 12)))
        table_header = QHBoxLayout()
        table_header.addWidget(label('Интервалы', 'section'))
        table_header.addStretch()
        self.export_btn = button('Сохранить CSV')
        self.export_btn.clicked.connect(self.export_csv)
        table_header.addWidget(self.export_btn)
        table_frame.layout().addLayout(table_header)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(['№', 'Левая', 'Правая', 'μ центра', 'Кластер', 'После α'])
        self.table.horizontalHeaderItem(4).setToolTip('Кластер в полном наборе')
        self.table.horizontalHeaderItem(5).setToolTip('Кластер после фильтрации по α')
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table_frame.layout().addWidget(self.table)
        right_layout.addWidget(table_frame, 4)
        split.addWidget(right)
        split.setSizes([288, 960])
        split.setStretchFactor(1, 1)
        self._refresh_results()

    def _metric_changed(self):
        self.full = self.filtered = None
        self._refresh_results()

    def set_filter(self, alpha: float, lam: float):
        self.alpha, self.lam = alpha, lam
        self.filtered = None
        self._refresh_results()

    def _refresh_results(self):
        mask = alpha_mask(self.samples, self.lam, self.alpha)
        self.cluster_chart.plot_clusters(self.samples, mask, self.metric.currentIndex(),
                                         self.full, self.filtered)
        self._refresh_table(mask)
        if self.samples:
            message = f'Прошли фильтр: {mask.sum()} из {len(self.samples)}'
            if mask.any() and self.filtered is None:
                message += '\nЗапустите кластеризацию для этого отбора.'
            self.cluster_status.setText(message)

    def generate_data(self):
        try:
            n = self.n_samples.value()
            seed = self.seed.value()
            self.samples = generate_samples(n, self.lam, seed=seed)
            self.full = self.filtered = None
            self._refresh_results()
            self._write_log(f'Сгенерировано {n} интервалов при λ={self.lam:g}, seed={seed}.')
            for index, number in enumerate(self.samples, 1):
                self.journal.info('X%d: [%.8g; %.8g].', index, number.left, number.right)
        except (ValueError, OverflowError) as exc:
            QMessageBox.warning(self, 'Ошибка генерации', str(exc))

    def _set_busy(self, busy: bool):
        for widget in (self.cluster_btn, self.generate_btn, self.metric,
                       self.k_count, self.iterations, self.seed):
            widget.setDisabled(busy)
        self.busy_changed.emit(busy)

    def cluster_data(self):
        if self.thread is not None:
            return
        if not self.samples:
            QMessageBox.information(self, 'Нет данных', 'Сначала сгенерируйте интервалы.')
            return
        self.cluster_status.setText('Вычисление кластеров…')
        self._set_busy(True)
        self.thread = QThread(self)
        self.worker = ClusteringWorker(
            self.samples, alpha_mask(self.samples, self.lam, self.alpha), self.k_count.value(),
            self.metric.currentIndex(), self.iterations.value(), self.seed.value(),
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.result.connect(self._cluster_success)
        self.worker.error.connect(self._cluster_error)
        self.worker.result.connect(self.thread.quit)
        self.worker.error.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self._worker_finished)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    @pyqtSlot(object)
    def _cluster_success(self, results):
        self.full, self.filtered = results
        self._refresh_results()
        self._write_log(f'k-medoids: k={self.k_count.value()}, метрика «{METRICS[self.metric.currentIndex()]}», '
                        f'λ={self.lam:g}, α={self.alpha:g}. {self.cluster_status.text()}')
        for idx, label_id in enumerate(self.full[0], 1):
            self.journal.info('X%d: кластер %d.', idx, label_id + 1)
        if self.filtered is not None:
            indices = np.flatnonzero(alpha_mask(self.samples, self.lam, self.alpha))
            for idx, label_id in zip(indices, self.filtered[0]):
                self.journal.info('X%d: кластер после α %d.', idx + 1, label_id + 1)

    @pyqtSlot(str)
    def _cluster_error(self, message: str):
        self.cluster_status.setText('Ошибка анализа — см. журнал.')
        self._write_log(f'Ошибка кластеризации: {message}')
        QMessageBox.warning(self, 'Ошибка кластеризации', message)

    @pyqtSlot()
    def _worker_finished(self):
        self.thread = None
        self.worker = None
        self._set_busy(False)

    def _refresh_table(self, mask):
        membership = exponential_membership([number.center for number in self.samples], self.lam)
        filtered_labels = self._filtered_labels(mask)
        self.table.setUpdatesEnabled(False)
        try:
            self.table.setRowCount(len(self.samples))
            for idx, number in enumerate(self.samples):
                full_id = '—' if self.full is None else str(self.full[0][idx] + 1)
                filtered_id = filtered_labels.get(idx, '—') if mask[idx] else 'Отсеян'
                values = (str(idx + 1), f'{number.left:.4f}', f'{number.right:.4f}',
                          f'{membership[idx]:.4f}', full_id, str(filtered_id))
                for col, value in enumerate(values):
                    cell = QTableWidgetItem(value)
                    cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    if col >= 4 and value.isdecimal():
                        cell.setForeground(QColor(CLUSTER_COLORS[(int(value) - 1) % len(CLUSTER_COLORS)]))
                    self.table.setItem(idx, col, cell)
        finally:
            self.table.setUpdatesEnabled(True)

    def _filtered_labels(self, mask):
        if self.filtered is None:
            return {}
        return dict(zip(np.flatnonzero(mask), self.filtered[0] + 1))

    def export_csv(self):
        if not self.samples:
            QMessageBox.information(self, 'Нет данных', 'Сначала сгенерируйте набор величин.')
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Сохранить набор', 'fuzzy_samples.csv',
                                              'CSV-файлы (*.csv)')
        if not path:
            return
        try:
            mask = alpha_mask(self.samples, self.lam, self.alpha)
            labels = None if self.full is None else self.full[0] + 1
            write_samples_csv(path, self.samples, self.lam, self.alpha, mask,
                              labels, self._filtered_labels(mask))
            self._write_log(f'Экспортировано {len(self.samples)} величин: {path}')
            QMessageBox.information(self, 'Готово', 'CSV-файл успешно сохранён.')
        except OSError as exc:
            QMessageBox.warning(self, 'Не удалось сохранить файл', str(exc))
