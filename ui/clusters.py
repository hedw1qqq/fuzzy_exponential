"""Генерация набора и фоновая кластеризация."""

from collections.abc import Callable

import numpy as np
from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QComboBox, QFileDialog, QFrame, QHBoxLayout, QHeaderView, QMessageBox,
    QRadioButton, QSplitter, QTableWidget, QTableWidgetItem, QWidget,
)

from clustering import Sample, cluster_numbers, generate_samples
from fuzzy_numbers import FuzzyNumber
from .storage import Journal, write_samples_csv
from .widgets import Chart, button, card, field_layout, float_box, int_box, label, make_scroll, vertical


class ClusteringWorker(QObject):
    result = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, numbers: list[FuzzyNumber], mode: str, value: float | int):
        super().__init__()
        self.numbers = numbers
        self.mode = mode
        self.value = value

    @pyqtSlot()
    def run(self):
        try:
            labels = cluster_numbers(
                self.numbers,
                k=int(self.value) if self.mode == 'count' else None,
                threshold=float(self.value) if self.mode == 'threshold' else None,
            )
            self.result.emit(labels)
        except Exception as exc:
            self.error.emit(str(exc))


class ClustersPage(QWidget):
    def __init__(self, write_log: Callable[[str], None], journal: Journal):
        super().__init__()
        self._write_log = write_log
        self.journal = journal
        self.samples: list[Sample] = []
        self.cluster_labels: np.ndarray | None = None
        self.worker: ClusteringWorker | None = None
        self.thread: QThread | None = None
        self.setObjectName('page')
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(split)
        panel = QWidget()
        panel.setObjectName('page')
        stack = vertical(12, (0, 0, 4, 0))
        panel.setLayout(stack)
        generator = card(vertical(12))
        g = generator.layout()
        g.addWidget(label('Генерация выборки', 'section'))
        self.n_samples = int_box(150, 100, 300)
        self.seed = int_box(42, 0, 2_147_483_647)
        self.generation_mode = QComboBox()
        self.generation_mode.addItems(['Группы (4 центра)', 'Равномерно'])
        g.addLayout(field_layout('Количество величин (100–300)', self.n_samples))
        g.addLayout(field_layout('Seed (для повторяемости)', self.seed))
        g.addLayout(field_layout('Распределение параметров', self.generation_mode))
        self.alpha = float_box(0.5, 0.01, 1.0, 2)
        g.addLayout(field_layout('Общий уровень α (один интервал на величину)', self.alpha))
        self.alpha.valueChanged.connect(self._alpha_changed)
        self.generate_btn = button('Сгенерировать набор', primary=True)
        self.generate_btn.clicked.connect(self.generate_data)
        g.addWidget(self.generate_btn)
        stack.addWidget(generator)

        settings = card(vertical(12))
        s = settings.layout()
        s.addWidget(label('Критерий кластеризации', 'section'))
        self.mode_count = QRadioButton('Заданное число кластеров')
        self.mode_threshold = QRadioButton('Порог расстояния')
        self.mode_count.setChecked(True)
        s.addWidget(self.mode_count)
        self.k_count = int_box(4, 1, 300)
        s.addLayout(field_layout('Количество k', self.k_count))
        s.addWidget(self.mode_threshold)
        self.threshold = float_box(0.9, 0.001, 1000.0, 3)
        s.addLayout(field_layout('Порог d', self.threshold))
        s.addWidget(label('Работает только выбранный режим: k или d.', 'hint', True))
        self.mode_count.toggled.connect(self.update_cluster_mode)
        self.mode_threshold.toggled.connect(self.update_cluster_mode)
        self.update_cluster_mode()
        self.cluster_btn = button('Выполнить кластеризацию', primary=True)
        self.cluster_btn.clicked.connect(self.cluster_data)
        s.addWidget(self.cluster_btn)
        self.export_btn = button('Экспорт таблицы CSV')
        self.export_btn.clicked.connect(self.export_csv)
        s.addWidget(self.export_btn)
        stack.addWidget(settings)
        info = QFrame()
        info.setObjectName('summary')
        info.setLayout(vertical(8, (15, 15, 15, 15)))
        info.layout().addWidget(label('ИТОГ АНАЛИЗА', 'hint'))
        self.cluster_status = label('Сначала сгенерируйте набор.', 'result', True)
        info.layout().addWidget(self.cluster_status)
        stack.addWidget(info)
        stack.addWidget(label('Алгоритм: иерархический complete linkage.\n'
                              'Метрика: RMS расстояние границ α-срезов.', 'hint', True))
        stack.addStretch(1)
        scroll = make_scroll(panel)
        scroll.setMinimumWidth(285)
        split.addWidget(scroll)

        right = QWidget()
        right.setObjectName('page')
        right_layout = vertical(10, (0, 0, 0, 0))
        right.setLayout(right_layout)
        chart_frame = card(vertical(7, (14, 13, 14, 13)))
        chart_frame.layout().addWidget(label('Пространство параметров', 'section'))
        self.cluster_chart = Chart()
        chart_frame.layout().addWidget(self.cluster_chart, 1)
        right_layout.addWidget(chart_frame, 3)
        table_frame = card(vertical(8, (14, 12, 14, 12)))
        table_frame.layout().addWidget(label('Нечёткие величины и принадлежность к кластерам', 'section'))
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(['№', 'Сдвиг a', 'Интенсивность λ', 'Масштаб 1/λ', 'Кластер'])
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table_frame.layout().addWidget(self.table)
        right_layout.addWidget(table_frame, 2)
        split.addWidget(right)
        split.setSizes([350, 830])
        split.setStretchFactor(1, 1)
        self.cluster_chart.plot_clusters(self.samples, self.cluster_labels)

    def _alpha_changed(self, _value: float):
        # Старые интервалы соответствуют предыдущему α. Нельзя смешивать их с новыми.
        self.samples = []
        self.cluster_labels = None
        self._refresh_table()
        self.cluster_chart.plot_clusters([], None)
        self.cluster_status.setText('Уровень α изменён. Сгенерируйте новый набор.')

    def update_cluster_mode(self, _checked: bool | None = None):
        self.k_count.setEnabled(self.mode_count.isChecked())
        self.threshold.setEnabled(self.mode_threshold.isChecked())

    def generate_data(self):
        try:
            n = self.n_samples.value()
            seed = self.seed.value()
            mode = 'groups' if self.generation_mode.currentIndex() == 0 else 'uniform'
            self.samples = generate_samples(n, seed=seed, mode=mode, alpha=self.alpha.value())
            self.cluster_labels = None
            self.cluster_chart.plot_clusters(self.samples, self.cluster_labels)
            self._refresh_table()
            self.cluster_status.setText(f'Создано интервалов: {n} (α={self.alpha.value():.2f})\n'
                                        f'Режим: {self.generation_mode.currentText()}\n'
                                        'Можно запускать кластеризацию.')
            self._write_log(f'Генерация {n} α-интервалов (α={self.alpha.value():.2f}), seed={seed}, mode={mode}.')
            # Записываем данные каждой величины в журнал на диске.
            for i, sample in enumerate(self.samples, 1):
                self.journal.info(f'X{i}: a={sample.a:.6f}, λ={sample.lam:.6f}, '
                                 f'α={sample.number.alpha:.2f} [{sample.number.left[0]:.6f}; {sample.number.right[0]:.6f}].')
        except (ValueError, OverflowError) as exc:
            QMessageBox.warning(self, 'Ошибка генерации', str(exc))

    def cluster_data(self):
        if not self.samples:
            QMessageBox.information(self, 'Нет данных', 'Сначала сгенерируйте набор величин.')
            return
        if self.thread is not None:
            return
        mode = 'count' if self.mode_count.isChecked() else 'threshold'
        value = self.k_count.value() if mode == 'count' else self.threshold.value()
        if mode == 'count' and value > len(self.samples):
            QMessageBox.warning(self, 'Неверное k', 'Число кластеров не должно превышать объём набора.')
            return
        self.current_condition = f'k={value}' if mode == 'count' else f'порог d={value:g}'
        self.cluster_status.setText('Вычисление кластеров…')
        self.cluster_btn.setEnabled(False)
        self.generate_btn.setEnabled(False)
        self.alpha.setEnabled(False)
        self.thread = QThread(self)
        self.worker = ClusteringWorker([s.number for s in self.samples], mode, value)
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
    def _cluster_success(self, labels):
        self.cluster_labels = np.asarray(labels, dtype=int)
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        sizes = ', '.join(str(x) for x in counts)
        self.cluster_status.setText(f'Объектов: {len(labels)} (α={self.samples[0].number.alpha:.2f})\n'
                                    f'Получено кластеров: {len(unique)}\n'
                                    f'Размеры: {sizes}')
        self._refresh_table()
        self.cluster_chart.plot_clusters(self.samples, self.cluster_labels)
        self._write_log(f'Кластеризация {self.current_condition}: '
                        f'{len(unique)} кластеров, размеры {counts.tolist()}.')
        for idx, cluster_id in enumerate(self.cluster_labels, 1):
            self.journal.info('X%d: кластер %d.', idx, cluster_id)

    @pyqtSlot(str)
    def _cluster_error(self, message: str):
        self.cluster_status.setText('Ошибка анализа — см. журнал.')
        self._write_log(f'Ошибка кластеризации: {message}')
        QMessageBox.warning(self, 'Ошибка кластеризации', message)

    @pyqtSlot()
    def _worker_finished(self):
        self.thread = None
        self.worker = None
        self.cluster_btn.setEnabled(True)
        self.generate_btn.setEnabled(True)
        self.alpha.setEnabled(True)

    def _refresh_table(self):
        self.table.setUpdatesEnabled(False)
        try:
            self.table.setRowCount(len(self.samples))
            for idx, item in enumerate(self.samples):
                cluster_id = ('—' if self.cluster_labels is None else str(self.cluster_labels[idx]))
                values = (str(idx + 1), f'{item.a:.4f}', f'{item.lam:.4f}',
                          f'{1.0 / item.lam:.4f}', cluster_id)
                for col, value in enumerate(values):
                    cell = QTableWidgetItem(value)
                    cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.table.setItem(idx, col, cell)
        finally:
            self.table.setUpdatesEnabled(True)

    def export_csv(self):
        if not self.samples:
            QMessageBox.information(self, 'Нет данных', 'Сначала сгенерируйте набор величин.')
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Сохранить набор', 'fuzzy_samples.csv',
                                              'CSV-файлы (*.csv)')
        if not path:
            return
        try:
            write_samples_csv(path, self.samples, self.cluster_labels)
            self._write_log(f'Экспортировано {len(self.samples)} величин: {path}')
            QMessageBox.information(self, 'Готово', 'CSV-файл успешно сохранён.')
        except OSError as exc:
            QMessageBox.warning(self, 'Не удалось сохранить файл', str(exc))
