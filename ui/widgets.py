"""Общие виджеты и графики."""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDoubleSpinBox, QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea,
    QSizePolicy, QSpinBox, QVBoxLayout, QWidget,
)

from clustering import Sample
from fuzzy_numbers import FuzzyNumber, exponential_membership


PLOT = '#131f32'
INK = '#e8eef9'
MUTED = '#91a1b9'
TEAL = '#4dd3b2'
PURPLE = '#ab9aff'
ORANGE = '#ffbf72'


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

    def plot_values(self, a, la, b, lb, A: FuzzyNumber, B: FuzzyNumber,
                    level: float, result: FuzzyNumber | None = None):
        ax, ax_cut = self.axes
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
        self.alpha_line = ax_cut.axhline(level, color='#c3d3e5', alpha=.8, lw=1.0, ls=':')
        ax_cut.set_ylim(0, 1.035)
        ax_cut.set_title('Границы α-срезов', loc='left', color=INK,
                         fontsize=12, fontweight='bold', pad=12)
        ax_cut.set_xlabel('Значение x')
        ax_cut.set_ylabel('Уровень α')
        legend2 = ax_cut.legend(loc='upper right', fontsize=8, ncol=2, framealpha=.12)
        for item in legend2.get_texts():
            item.set_color(INK)
        self.draw_idle()

    def set_alpha(self, level: float):
        self.alpha_line.set_ydata([level, level])
        self.draw_idle()

    def plot_clusters(self, samples: list[Sample], labels: np.ndarray | None):
        ax = self.axes[0]
        ax.clear()
        Chart.style_axis(ax)
        ax.set_title('Положение и масштаб нечётких величин', loc='left', color=INK,
                     fontsize=12, fontweight='bold', pad=14)
        ax.set_xlabel('Положение a')
        ax.set_ylabel('Масштаб 1/λ')
        if samples:
            xs = np.array([s.a for s in samples])
            ys = np.array([1 / s.lam for s in samples])
            if labels is None:
                ax.scatter(xs, ys, s=35, c=TEAL, alpha=.85, edgecolors='none')
            else:
                ids = np.unique(labels)
                if len(ids) <= 12:
                    colors = ('#4dd3b2', '#ab9aff', '#ffbf72', '#ef7d8e',
                              '#71c9ff', '#e5ca7c', '#ff94d1', '#79e5a0',
                              '#b7aafc', '#8ee7e9', '#e7a879', '#83a1f4')
                    for i, group in enumerate(ids):
                        mask = labels == group
                        ax.scatter(xs[mask], ys[mask], s=36, color=colors[i % len(colors)],
                                   label=f'Кластер {group} · {mask.sum()}', alpha=.9, edgecolors='none')
                    legend = ax.legend(loc='upper right', framealpha=.12, fontsize=8, ncol=2)
                    for item in legend.get_texts():
                        item.set_color(INK)
                else:
                    ax.scatter(xs, ys, s=32, c=labels, cmap='tab20',
                               alpha=.9, edgecolors='none')
        else:
            ax.text(.5, .5, 'Сгенерируйте данные, чтобы увидеть график',
                    transform=ax.transAxes, ha='center', va='center', color=MUTED)
        self.draw_idle()
