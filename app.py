"""Графический интерфейс PyQt6 для экспоненциальных нечётких величин.

Математические алгоритмы находятся в fuzzy_numbers.py и clustering.py.
Запуск: python app.py
"""

from __future__ import annotations

import csv
from datetime import datetime
import logging
import sys
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QButtonGroup, QComboBox, QDoubleSpinBox, QFileDialog,
    QFrame, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPlainTextEdit, QPushButton, QRadioButton, QScrollArea, QSlider,
    QSpinBox, QSplitter, QStackedWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget, QHeaderView, QSizePolicy,
)

from clustering import Sample, cluster_numbers, generate_samples
from fuzzy_numbers import FuzzyNumber, calculate, compare, exponential, exponential_membership


ROOT = Path(__file__).resolve().parent
BG = '#0b1220'
SIDEBAR = '#101a2b'
CARD = '#172338'
PLOT = '#131f32'
INK = '#e8eef9'
MUTED = '#91a1b9'
LINE = '#29374c'
TEAL = '#4dd3b2'
PURPLE = '#ab9aff'
ORANGE = '#ffbf72'

STYLE = f"""
QMainWindow, QWidget#root, QWidget#page {{ background: {BG}; color: {INK}; }}
QFrame#sidebar {{ background: {SIDEBAR}; border-right: 1px solid {LINE}; }}
QLabel {{ color: {INK}; background: transparent; }}
QLabel#brand {{ color: #f7faff; font-size: 18px; font-weight: 800; }}
QLabel#subbrand {{ color: {MUTED}; font-size: 11px; }}
QLabel#title {{ font-size: 26px; font-weight: 750; color: #f6f8ff; }}
QLabel#subtitle {{ color: {MUTED}; font-size: 12px; }}
QLabel#section {{ font-size: 15px; font-weight: 700; color: #eff3fd; }}
QLabel#hint {{ color: {MUTED}; font-size: 11px; }}
QLabel#result {{ color: #c8f7e8; font-size: 12px; line-height: 1.4; }}
QFrame#card {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 13px; }}
QFrame#summary {{ background: #152f36; border: 1px solid #27635b; border-radius: 10px; }}
QPushButton {{ color: {INK}; background: #26364d; border: 1px solid #31435f; border-radius: 9px; padding: 11px 14px; font-size: 12px; font-weight: 650; }}
QPushButton:hover {{ background: #344b65; }}
QPushButton:pressed {{ background: #1f3048; }}
QPushButton:disabled {{ background: #1a2739; color: #65758d; border-color: #26354b; }}
QPushButton#primary {{ color: #07251f; background: {TEAL}; border: none; font-weight: 800; }}
QPushButton#primary:hover {{ background: #7ce5ca; }}
QPushButton#nav {{ color: #a9b9d0; background: transparent; border: none; border-radius: 10px; text-align: left; padding: 14px 16px; font-weight: 650; font-size: 13px; }}
QPushButton#nav:hover {{ background: #20314b; color: #eff6ff; }}
QPushButton#nav:checked {{ background: #1f3e46; color: #83f2d2; border-left: 3px solid {TEAL}; padding-left: 13px; }}
QDoubleSpinBox, QSpinBox, QComboBox {{ background: #101b2d; color: {INK}; border: 1px solid #34455d; border-radius: 8px; padding: 9px 12px; min-height: 20px; selection-background-color: #296d65; }}
QDoubleSpinBox:hover, QSpinBox:hover, QComboBox:hover {{ border-color: #60aa9c; }}
QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {{ border: 1px solid {TEAL}; }}
QComboBox QAbstractItemView {{ background: #172538; color: {INK}; selection-background-color: #2b5b59; }}
QRadioButton {{ color: {INK}; spacing: 7px; background: transparent; }}
QRadioButton::indicator {{ width: 15px; height: 15px; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ border: none; background: #111c2b; width: 10px; margin: 3px 1px; }}
QScrollBar::handle:vertical {{ background: #41516a; border-radius: 4px; min-height: 28px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ height: 9px; background: #111c2b; }}
QScrollBar::handle:horizontal {{ background: #41516a; border-radius: 4px; min-width: 26px; }}
QSplitter::handle {{ background: transparent; width: 9px; }}
QTableWidget {{ background: #101c2c; alternate-background-color: #17263b; color: {INK}; border: 1px solid {LINE}; border-radius: 8px; gridline-color: #25374e; selection-background-color: #286058; selection-color: #ffffff; }}
QHeaderView::section {{ background: #27394e; color: #b8d0e4; padding: 9px 6px; border: none; border-right: 1px solid #34465a; font-weight: 700; }}
QPlainTextEdit {{ background: #0e1a29; color: #cfe1f2; border: 1px solid {LINE}; border-radius: 9px; padding: 14px; selection-background-color: #24655e; }}
QSlider::groove:horizontal {{ height: 5px; background: #395069; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {TEAL}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: #bafbed; border: 2px solid {TEAL}; width: 17px; margin: -7px 0; border-radius: 10px; }}
"""


def label(text: str, name: str | None = None, wrap: bool = False) -> QLabel:
    widget = QLabel(text)
    if name:
        widget.setObjectName(name)
    widget.setWordWrap(wrap)
    return widget


def button(text: str, *, primary: bool = False) -> QPushButton:
    widget = QPushButton(text)
    widget.setCursor(Qt.CursorShape.PointingHandCursor)
    if primary:
        widget.setObjectName('primary')
    return widget


def card(layout: QVBoxLayout | QHBoxLayout | None = None) -> QFrame:
    widget = QFrame()
    widget.setObjectName('card')
    if layout is not None:
        widget.setLayout(layout)
    return widget


def vertical(spacing: int = 12, margins=(18, 18, 18, 18)) -> QVBoxLayout:
    result = QVBoxLayout()
    result.setContentsMargins(*margins)
    result.setSpacing(spacing)
    return result


def make_scroll(widget: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setWidget(widget)
    return area


def field_layout(caption: str, widget: QWidget, hint: str | None = None) -> QVBoxLayout:
    box = vertical(5, (0, 0, 0, 0))
    box.addWidget(label(caption))
    box.addWidget(widget)
    if hint:
        box.addWidget(label(hint, 'hint', True))
    return box


def float_box(value: float, minimum: float, maximum: float, decimals: int = 3) -> QDoubleSpinBox:
    widget = QDoubleSpinBox()
    widget.setRange(minimum, maximum)
    widget.setDecimals(decimals)
    widget.setSingleStep(0.1)
    widget.setValue(value)
    return widget


def int_box(value: int, minimum: int, maximum: int) -> QSpinBox:
    widget = QSpinBox()
    widget.setRange(minimum, maximum)
    widget.setValue(value)
    return widget


class Chart(FigureCanvas):
    """Встраиваемый холст Matplotlib с единой тёмной темой."""
    def __init__(self, rows: int = 1, parent: QWidget | None = None):
        fig = Figure(figsize=(9, 6), dpi=100, facecolor=PLOT)
        super().__init__(fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        if rows == 2:
            self.axes = list(fig.subplots(2, 1))
            fig.subplots_adjust(left=.09, right=.97, top=.92, bottom=.11, hspace=.47)
        else:
            self.axes = [fig.add_subplot(111)]
            fig.subplots_adjust(left=.09, right=.97, top=.9, bottom=.16)
        for ax in self.axes:
            self.style_axis(ax)

    @staticmethod
    def style_axis(ax) -> None:
        ax.set_facecolor(PLOT)
        ax.tick_params(colors='#a7b5cd', labelsize=9)
        for spine in ax.spines.values():
            spine.set_color('#41516a')
        ax.xaxis.label.set_color('#b6c4d9')
        ax.yaxis.label.set_color('#b6c4d9')
        ax.title.set_color(INK)
        ax.grid(color='#64738b', alpha=.20, linewidth=.7)


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


class FuzzyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Экспоненциальные нечёткие величины')
        self.resize(1380, 860)
        self.setMinimumSize(950, 660)
        self.samples: list[Sample] = []
        self.cluster_labels: np.ndarray | None = None
        self.plotted_numbers: list[FuzzyNumber] = []
        self.worker: ClusteringWorker | None = None
        self.thread: QThread | None = None
        self._setup_logger()
        self._build_ui()
        self.show_values()
        self._write_log('Приложение запущено.')

    def _setup_logger(self):
        try:
            log_dir = ROOT / 'logs'
            log_dir.mkdir(exist_ok=True)
        except OSError:
            log_dir = Path.home() / '.fuzzy_exponential_pyqt6'
            log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / 'app.log'
        self.logger = logging.getLogger('fuzzy_exponential_pyqt6')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
        handler = logging.FileHandler(self.log_path, encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s', '%Y-%m-%d %H:%M:%S'))
        self.logger.addHandler(handler)

    def _write_log(self, message: str):
        self.logger.info(message)
        if hasattr(self, 'log_view'):
            self.log_view.appendPlainText(f'[{datetime.now():%H:%M:%S}] {message}')
        if hasattr(self, 'status_text'):
            self.status_text.setText(message)

    def _build_ui(self):
        root = QWidget()
        root.setObjectName('root')
        self.setCentralWidget(root)
        shell = QHBoxLayout(root)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)

        sidebar = QFrame()
        sidebar.setObjectName('sidebar')
        sidebar.setFixedWidth(235)
        side = vertical(12, (20, 27, 20, 20))
        sidebar.setLayout(side)
        side.addWidget(label('Практическая работа', 'brand', True))
        side.addWidget(label('Экспоненциальные нечёткие величины', 'subbrand', True))
        side.addSpacing(25)
        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.nav_buttons = []
        for idx, title in enumerate(('Величины и операции', 'Кластеризация', 'Журнал', 'О модели')):
            nav = button(title)
            nav.setObjectName('nav')
            nav.setCheckable(True)
            nav.clicked.connect(lambda _checked=False, i=idx: self.open_page(i))
            self.nav_group.addButton(nav, idx)
            self.nav_buttons.append(nav)
            side.addWidget(nav)
        side.addStretch(1)
        side.addWidget(label('МАИ · Неклассические логики', 'hint', True))
        shell.addWidget(sidebar)

        content = QWidget()
        content.setObjectName('page')
        outer = vertical(14, (23, 18, 23, 16))
        content.setLayout(outer)
        header = QHBoxLayout()
        header_text = vertical(4, (0, 0, 0, 0))
        self.page_title = label('', 'title')
        self.page_subtitle = label('', 'subtitle')
        header_text.addWidget(self.page_title)
        header_text.addWidget(self.page_subtitle)
        header.addLayout(header_text)
        header.addStretch(1)
        outer.addLayout(header)
        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_values_page())
        self.pages.addWidget(self._build_cluster_page())
        self.pages.addWidget(self._build_journal_page())
        self.pages.addWidget(self._build_info_page())
        outer.addWidget(self.pages, 1)
        bottom = QHBoxLayout()
        self.status_text = label('Готово к работе.', 'hint', True)
        bottom.addWidget(self.status_text, 1)
        bottom.addWidget(label('μ(x) = exp(−λ(x−a)), x ≥ a', 'hint'))
        outer.addLayout(bottom)
        shell.addWidget(content, 1)
        self.open_page(0)

    def open_page(self, index: int):
        titles = [
            ('Величины и операции', 'Параметры, графики принадлежности, α-срезы и арифметика'),
            ('Кластеризация', 'Генерация 100–300 величин и кластерный анализ по двум критериям'),
            ('Журнал', 'История действий программы с сохранением в файл'),
            ('О математической модели', 'Формулы, принятые соглашения и ограничения'),
        ]
        self.pages.setCurrentIndex(index)
        self.nav_buttons[index].setChecked(True)
        self.page_title.setText(titles[index][0])
        self.page_subtitle.setText(titles[index][1])

    def _build_values_page(self):
        page = QWidget()
        page.setObjectName('page')
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        split = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(split)

        controls_widget = QWidget()
        controls_widget.setObjectName('page')
        controls = vertical(12, (0, 0, 4, 0))
        controls_widget.setLayout(controls)
        params = card(vertical(13))
        form = params.layout()
        form.addWidget(label('Параметры двух величин', 'section'))
        form.addWidget(label('a — положение, λ — скорость затухания', 'hint', True))
        form.addWidget(label('Величина A', 'section'))
        self.a_a = float_box(2.0, -100, 100)
        self.l_a = float_box(1.1, 0.01, 100)
        form.addLayout(field_layout('Сдвиг a', self.a_a))
        form.addLayout(field_layout('Интенсивность λ (> 0)', self.l_a))
        form.addWidget(label('Величина B', 'section'))
        self.a_b = float_box(5.0, -100, 100)
        self.l_b = float_box(1.7, 0.01, 100)
        form.addLayout(field_layout('Сдвиг a', self.a_b))
        form.addLayout(field_layout('Интенсивность λ (> 0)', self.l_b))
        form.addWidget(label('Операция', 'section'))
        self.operator = QComboBox()
        self.operator.addItems(['Сложение   A + B', 'Вычитание   A − B', 'Умножение   A × B', 'Деление   A ÷ B'])
        form.addWidget(self.operator)
        self.plot_button = button('Построить A и B', primary=True)
        self.plot_button.clicked.connect(self.show_values)
        form.addWidget(self.plot_button)
        operations = QHBoxLayout()
        self.calc_button = button('Вычислить')
        self.calc_button.clicked.connect(self.run_operation)
        self.compare_button = button('Сравнить')
        self.compare_button.clicked.connect(self.run_comparison)
        operations.addWidget(self.calc_button)
        operations.addWidget(self.compare_button)
        form.addLayout(operations)
        controls.addWidget(params)

        result_panel = QFrame()
        result_panel.setObjectName('summary')
        rlayout = vertical(8, (15, 15, 15, 15))
        result_panel.setLayout(rlayout)
        rlayout.addWidget(label('РЕЗУЛЬТАТ', 'hint'))
        self.operation_status = label('Задайте параметры и выберите действие.', 'result', True)
        rlayout.addWidget(self.operation_status)
        controls.addWidget(result_panel)
        cuts_card = card(vertical(7))
        cuts = cuts_card.layout()
        cuts.addWidget(label('Просмотр α-среза', 'section'))
        self.alpha_label = label('α = 0.50', 'hint')
        cuts.addWidget(self.alpha_label)
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setRange(1, 100)
        self.alpha_slider.setValue(50)
        self.alpha_slider.valueChanged.connect(self.update_alpha_cuts)
        cuts.addWidget(self.alpha_slider)
        self.alpha_result = label('', 'hint', True)
        cuts.addWidget(self.alpha_result)
        controls.addWidget(cuts_card)
        controls.addStretch(1)
        left = make_scroll(controls_widget)
        left.setMinimumWidth(285)
        split.addWidget(left)

        chart_frame = card(vertical(8, (15, 12, 15, 12)))
        chart_layout = chart_frame.layout()
        chart_layout.addWidget(label('Графики функций и границ α-срезов', 'section'))
        self.values_chart = Chart(rows=2)
        chart_layout.addWidget(self.values_chart, 1)
        chart_layout.addWidget(label('Пунктир — правая граница α-среза; сплошная — левая.', 'hint'))
        split.addWidget(chart_frame)
        split.setSizes([350, 830])
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        return page

    def _read_values(self):
        a, lam_a = self.a_a.value(), self.l_a.value()
        b, lam_b = self.a_b.value(), self.l_b.value()
        A = exponential(a, lam_a, 'A')
        B = exponential(b, lam_b, 'B')
        return a, lam_a, b, lam_b, A, B

    def _plot_values(self, a, la, b, lb, A: FuzzyNumber, B: FuzzyNumber,
                     result: FuzzyNumber | None = None):
        ax, ax_cut = self.values_chart.axes
        for axis in (ax, ax_cut):
            axis.clear()
            Chart.style_axis(axis)
        numbers = [A, B] + ([result] if result is not None else [])
        low = min(float(v.left[0]) for v in numbers)
        high = max(float(v.right[0]) for v in numbers)
        pad = max(.06 * (high - low), .25)
        xs = np.linspace(low - pad, high + pad, 1200)
        anchors = [xs, [np.nextafter(a, -np.inf), a, np.nextafter(b, -np.inf), b]]
        if result is not None:
            anchors.extend((result.left, result.right))
        xs = np.unique(np.concatenate(anchors))
        ax.plot(xs, exponential_membership(xs, a, la), lw=2.25, color=TEAL, label=f'A (a={a:g}, λ={la:g})')
        ax.plot(xs, exponential_membership(xs, b, lb), lw=2.25, color=PURPLE, label=f'B (a={b:g}, λ={lb:g})')
        items = [(A, TEAL, 'A'), (B, PURPLE, 'B')]
        if result is not None:
            ax.plot(xs, result.membership(xs), color=ORANGE, lw=2.1, label='Результат')
            items.append((result, ORANGE, 'C'))
        ax.set_title('Функции принадлежности μ(x)', loc='left', color=INK,
                     fontsize=12, fontweight='bold', pad=14)
        ax.set_xlabel('Значение x')
        ax.set_ylabel('Степень принадлежности')
        ax.set_ylim(-.04, 1.08)
        legend = ax.legend(loc='upper right', fontsize=8, framealpha=.12)
        for item in legend.get_texts():
            item.set_color(INK)
        for value, color, name in items:
            ax_cut.plot(value.left, value.alphas, color=color, lw=1.9, label=f'{name} · левый')
            ax_cut.plot(value.right, value.alphas, color=color, lw=1.9, ls='--', label=f'{name} · правый')
            ax_cut.fill_betweenx(value.alphas, value.left, value.right, color=color, alpha=.06)
        level = self.alpha_slider.value() / 100
        marker = ax_cut.axhline(level, color='#c3d3e5', alpha=.8, lw=1.0, ls=':')
        marker.set_gid('alpha-selection')
        ax_cut.set_ylim(0, 1.035)
        ax_cut.set_title('Границы α-срезов', loc='left', color=INK,
                         fontsize=12, fontweight='bold', pad=12)
        ax_cut.set_xlabel('Значение x')
        ax_cut.set_ylabel('Уровень α')
        legend2 = ax_cut.legend(loc='upper right', fontsize=8, ncol=2, framealpha=.12)
        for item in legend2.get_texts():
            item.set_color(INK)
        self.plotted_numbers = numbers
        self.values_chart.draw_idle()
        self.update_alpha_cuts(redraw=False)

    def update_alpha_cuts(self, _value: int | None = None, *, redraw: bool = True):
        level = self.alpha_slider.value() / 100
        self.alpha_label.setText(f'α = {level:.2f}')
        parts = []
        for name, number in zip(('A', 'B', 'C'), self.plotted_numbers):
            left = np.interp(level, number.alphas, number.left)
            right = np.interp(level, number.alphas, number.right)
            parts.append(f'{name}α: [{left:.3f}; {right:.3f}]')
        self.alpha_result.setText('\n'.join(parts))
        if redraw:
            self._plot_alpha_line(self.values_chart.axes[1], level)

    def _plot_alpha_line(self, ax, level):
        # Не изменяем основную серию данных, только перемещаем линию уровня.
        for line in list(ax.lines):
            if line.get_gid() == 'alpha-selection':
                line.remove()
        marker = ax.axhline(level, color='#d3e0ef', alpha=.82, lw=1.2, ls=':')
        marker.set_gid('alpha-selection')
        self.values_chart.draw_idle()

    def show_values(self):
        try:
            a, la, b, lb, A, B = self._read_values()
            self._plot_values(a, la, b, lb, A, B)
            self.operation_status.setText(f'A: a={a:g}, λ={la:g}\nB: a={b:g}, λ={lb:g}\nГрафики построены.')
            self._write_log(f'Построены A(a={a:g}, λ={la:g}) и B(a={b:g}, λ={lb:g}).')
        except (ValueError, OverflowError) as exc:
            QMessageBox.warning(self, 'Неверные параметры', str(exc))

    def run_operation(self):
        try:
            a, la, b, lb, A, B = self._read_values()
            op = ('+', '-', '*', '/')[self.operator.currentIndex()]
            C = calculate(A, B, op)
            self._plot_values(a, la, b, lb, A, B, C)
            message = (f'A {op} B\n'
                       f'α = 1: [{C.left[-1]:.3f}; {C.right[-1]:.3f}]\n'
                       f'α = 0.01: [{C.left[0]:.3f}; {C.right[0]:.3f}]\n'
                       f'Числовая оценка: {C.representative:.3f}')
            self.operation_status.setText(message)
            self._write_log(f'A(a={a:g}, λ={la:g}) {op} B(a={b:g}, λ={lb:g}): '
                            f'α=1 [{C.left[-1]:.5g}; {C.right[-1]:.5g}], '
                            f'α=0.01 [{C.left[0]:.5g}; {C.right[0]:.5g}], '
                            f'оценка {C.representative:.5g}.')
        except (ValueError, OverflowError) as exc:
            QMessageBox.warning(self, 'Операция не выполнена', str(exc))
            self._write_log(f'Ошибка операции: {exc}')

    def run_comparison(self):
        try:
            a, la, b, lb, A, B = self._read_values()
            sign = compare(A, B)
            self._plot_values(a, la, b, lb, A, B)
            self.operation_status.setText(f'A {sign} B\n'
                                          f'Оценка A: {A.representative:.3f}\n'
                                          f'Оценка B: {B.representative:.3f}\n'
                                          'Сравнение по серединам α-срезов.')
            self._write_log(f'Сравнение A(a={a:g}, λ={la:g}) {sign} B(a={b:g}, λ={lb:g}): '
                            f'оценки {A.representative:.5g} и {B.representative:.5g}.')
        except (ValueError, OverflowError) as exc:
            QMessageBox.warning(self, 'Сравнение не выполнено', str(exc))

    def _build_cluster_page(self):
        page = QWidget()
        page.setObjectName('page')
        layout = QHBoxLayout(page)
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
        self._draw_clusters()
        return page

    def update_cluster_mode(self, _checked: bool | None = None):
        if hasattr(self, 'k_count'):
            self.k_count.setEnabled(self.mode_count.isChecked())
            self.threshold.setEnabled(self.mode_threshold.isChecked())

    def generate_data(self):
        try:
            n = self.n_samples.value()
            seed = self.seed.value()
            mode = 'groups' if self.generation_mode.currentIndex() == 0 else 'uniform'
            self.samples = generate_samples(n, seed=seed, mode=mode)
            self.cluster_labels = None
            self._draw_clusters()
            self._refresh_table()
            self.cluster_status.setText(f'Создано величин: {n}\n'
                                        f'Режим: {self.generation_mode.currentText()}\n'
                                        'Можно запускать кластеризацию.')
            self._write_log(f'Генерация {n} нечётких величин, seed={seed}, mode={mode}.')
            # Записываем данные каждой величины в журнал на диске.
            for i, sample in enumerate(self.samples, 1):
                self.logger.info(f'X{i}: a={sample.a:.6f}, λ={sample.lam:.6f}, '
                                 f'α=0.01 [{sample.number.left[0]:.6f}; {sample.number.right[0]:.6f}].')
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
        self.cluster_status.setText(f'Объектов: {len(labels)}\n'
                                    f'Получено кластеров: {len(unique)}\n'
                                    f'Размеры: {sizes}')
        self._refresh_table()
        self._draw_clusters()
        self._write_log(f'Кластеризация {self.current_condition}: '
                        f'{len(unique)} кластеров, размеры {counts.tolist()}.')
        for idx, cluster_id in enumerate(self.cluster_labels, 1):
            self.logger.info('X%d: кластер %d.', idx, cluster_id)

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

    def _draw_clusters(self):
        ax = self.cluster_chart.axes[0]
        ax.clear()
        Chart.style_axis(ax)
        ax.set_title('Положение и масштаб нечётких величин', loc='left', color=INK,
                     fontsize=12, fontweight='bold', pad=14)
        ax.set_xlabel('Положение a')
        ax.set_ylabel('Масштаб 1/λ')
        if self.samples:
            xs = np.array([s.a for s in self.samples])
            ys = np.array([1 / s.lam for s in self.samples])
            if self.cluster_labels is None:
                ax.scatter(xs, ys, s=35, c=TEAL, alpha=.85, edgecolors='none')
            else:
                ids = np.unique(self.cluster_labels)
                if len(ids) <= 12:
                    colors = ('#4dd3b2', '#ab9aff', '#ffbf72', '#ef7d8e',
                              '#71c9ff', '#e5ca7c', '#ff94d1', '#79e5a0',
                              '#b7aafc', '#8ee7e9', '#e7a879', '#83a1f4')
                    for i, group in enumerate(ids):
                        mask = self.cluster_labels == group
                        ax.scatter(xs[mask], ys[mask], s=36, color=colors[i % len(colors)],
                                   label=f'Кластер {group} · {mask.sum()}', alpha=.9, edgecolors='none')
                    legend = ax.legend(loc='upper right', framealpha=.12, fontsize=8, ncol=2)
                    for item in legend.get_texts():
                        item.set_color(INK)
                else:
                    ax.scatter(xs, ys, s=32, c=self.cluster_labels, cmap='tab20',
                               alpha=.9, edgecolors='none')
        else:
            ax.text(.5, .5, 'Сгенерируйте данные, чтобы увидеть график',
                    transform=ax.transAxes, ha='center', va='center', color=MUTED)
        self.cluster_chart.draw_idle()

    def export_csv(self):
        if not self.samples:
            QMessageBox.information(self, 'Нет данных', 'Сначала сгенерируйте набор величин.')
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Сохранить набор', 'fuzzy_samples.csv',
                                              'CSV-файлы (*.csv)')
        if not path:
            return
        try:
            with open(path, 'w', newline='', encoding='utf-8-sig') as fh:
                writer = csv.writer(fh, delimiter=';')
                writer.writerow(['id', 'a', 'lambda', 'scale_1/lambda',
                                 'alpha_min', 'left_min', 'right_min', 'cluster'])
                for idx, sample in enumerate(self.samples):
                    writer.writerow([idx + 1, f'{sample.a:.8f}', f'{sample.lam:.8f}',
                                     f'{1/sample.lam:.8f}', f'{sample.number.alphas[0]:.5f}',
                                     f'{sample.number.left[0]:.8f}', f'{sample.number.right[0]:.8f}',
                                     '' if self.cluster_labels is None else int(self.cluster_labels[idx])])
            self._write_log(f'Экспортировано {len(self.samples)} величин: {path}')
            QMessageBox.information(self, 'Готово', 'CSV-файл успешно сохранён.')
        except OSError as exc:
            QMessageBox.warning(self, 'Не удалось сохранить файл', str(exc))

    def _build_journal_page(self):
        page = QWidget()
        page.setObjectName('page')
        outer = vertical(14, (0, 0, 0, 0))
        page.setLayout(outer)
        surface = card(vertical(14))
        top = QHBoxLayout()
        top.addWidget(label('История действий и ошибок', 'section'))
        top.addStretch(1)
        clear = button('Очистить окно')
        clear.clicked.connect(self.clear_log)
        save = button('Сохранить журнал', primary=True)
        save.clicked.connect(self.save_log)
        top.addWidget(clear)
        top.addWidget(save)
        surface.layout().addLayout(top)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.document().setMaximumBlockCount(2500)
        self.log_view.setFont(QFont('Consolas', 10))
        try:
            history = self.log_path.read_text(encoding='utf-8').splitlines()[-700:]
            self.log_view.setPlainText('\n'.join(history))
        except OSError:
            pass
        surface.layout().addWidget(self.log_view, 1)
        surface.layout().addWidget(label(f'Автоматический файл: {self.log_path}', 'hint', True))
        outer.addWidget(surface)
        return page

    def clear_log(self):
        self.log_view.clear()
        self._write_log('Окно журнала очищено. История на диске сохранена.')

    def save_log(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Сохранить журнал', 'fuzzy_log.txt',
                                              'Журнал (*.txt *.log)')
        if not path:
            return
        try:
            for handler in self.logger.handlers:
                handler.flush()
            Path(path).write_text(self.log_path.read_text(encoding='utf-8'), encoding='utf-8')
            self._write_log(f'Сохранена копия журнала: {path}')
            QMessageBox.information(self, 'Готово', 'Журнал сохранён.')
        except OSError as exc:
            QMessageBox.warning(self, 'Ошибка записи', str(exc))

    def _build_info_page(self):
        page = QWidget()
        page.setObjectName('page')
        content = QWidget()
        content.setObjectName('page')
        body = vertical(14)
        content.setLayout(body)
        sections = [
            ('01  Экспоненциальная модель',
             'Для x < a принадлежность равна 0. При x ≥ a функция μ(x) = exp(−λ(x−a)), λ > 0. '
             'Это плотность сдвинутого экспоненциального распределения, нормированная на её максимум λ. '
             'Вариант «экспоненциальное распределение» здесь интерпретируется как нечёткая величина.'),
            ('02  α-срезы',
             'Aα = [a, a − ln(α)/λ], 0 < α ≤ 1. Компьютер хранит 199 срезов: от α=0.01 до α=1. '
             'При α→0 носитель справа неограничен, поэтому результаты являются численным приближением.'),
            ('03  Операции и сравнение',
             'Сложение/вычитание: арифметика границ интервалов. Умножение/деление: min/max '
             'попарных комбинаций концов интервалов. Деление запрещено, если интервал делителя '
             'содержит ноль. Сравнение выполняется по среднему значению середин α-срезов; '
             'это выбранный критерий, а не универсальный порядок нечётких множеств.'),
            ('04  Кластерный анализ',
             'Генерация 100–300 величин с параметрами (a, λ). Расстояние — RMS отклонение левых и '
             'правых границ α-срезов. Агломеративная кластеризация complete linkage останавливается '
             'либо при k кластерах, либо при превышении выбранного порога расстояния.'),
            ('05  Структура и использование',
             'fuzzy_numbers.py — независимая библиотека математических операций; '
             'clustering.py — библиотека генерации и кластеризации; app.py — только интерфейс PyQt6. '
             'Журнал автоматически создаётся в logs/app.log, результаты можно выгрузить в CSV. '
             'Подробности, ограничения и инструкция запуска — в README.md.'),
        ]
        for heading, desc in sections:
            panel = card(vertical(10))
            panel.layout().addWidget(label(heading, 'section'))
            panel.layout().addWidget(label(desc, None, True))
            body.addWidget(panel)
        body.addStretch(1)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(make_scroll(content))
        return page

    def closeEvent(self, event):
        if self.thread is not None and self.thread.isRunning():
            QMessageBox.information(self, 'Идёт вычисление',
                                    'Завершите кластеризацию перед закрытием приложения.')
            event.ignore()
            return
        self._write_log('Приложение закрыто.')
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setFont(QFont('Segoe UI', 10))
    app.setStyleSheet(STYLE)
    window = FuzzyWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
