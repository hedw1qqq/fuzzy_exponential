"""Общие виджеты и графики."""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDoubleSpinBox, QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea,
    QSizePolicy, QSpinBox, QVBoxLayout, QWidget,
)

from fuzzy_numbers import FuzzyNumber, exponential_membership


PLOT = '#151d29'
INK = '#ecf3fb'
MUTED = '#a9bace'
TEAL = '#57dfcc'
PURPLE = '#a7b8ff'
ORANGE = '#ffc27a'
CLUSTER_COLORS = (TEAL, PURPLE, ORANGE, '#f4a9db', '#84c7fa', '#c2dc81', '#d3b1fa', '#fba6a0')


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
    caption_label = label(caption, 'field')
    caption_label.setBuddy(widget)
    box.addWidget(caption_label)
    if not widget.accessibleName():
        widget.setAccessibleName(caption)
    box.addWidget(widget)
    if hint:
        box.addWidget(label(hint, 'hint', True))
    return box


def float_box(value: float, minimum: float, maximum: float, decimals: int = 3) -> QDoubleSpinBox:
    widget = QDoubleSpinBox()
    widget.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
    widget.setRange(minimum, maximum)
    widget.setDecimals(decimals)
    widget.setSingleStep(0.1)
    widget.setValue(value)
    return widget


def int_box(value: int, minimum: int, maximum: int) -> QSpinBox:
    widget = QSpinBox()
    widget.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
    widget.setRange(minimum, maximum)
    widget.setValue(value)
    return widget


class Chart(FigureCanvas):
    """Встраиваемый холст для графиков приложения."""
    def __init__(self, *, side_by_side: bool = False):
        fig = Figure(figsize=(9, 6), dpi=100, facecolor=PLOT, layout='constrained')
        super().__init__(fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        if side_by_side:
            self.axes = list(fig.subplots(1, 2))
        else:
            self.axes = list(fig.subplots(2, 1, sharex=True, height_ratios=(2.1, 1)))
        self.figure.get_layout_engine().set(h_pad=.10, w_pad=.10, hspace=.08, wspace=.12)

    @staticmethod
    def style_axis(ax) -> None:
        ax.set_facecolor(PLOT)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        for side, spine in ax.spines.items():
            spine.set_color('#475974')
            spine.set_visible(side in ('left', 'bottom'))
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.title.set_color(INK)
        ax.set_axisbelow(True)
        ax.grid(color='#303e54', linewidth=.65)

    def plot_values(self, numbers: tuple[FuzzyNumber, ...], membership: np.ndarray,
                    lam: float, alpha: float):
        ax, intervals = self.axes
        for axis in self.axes:
            axis.clear()
            self.style_axis(axis)
        low = min(0, *(value.left for value in numbers))
        high = max(3 / lam, *(value.right for value in numbers))
        pad = max(.06 * (high - low), .25)
        xs = np.linspace(low - pad, high + pad, 1200)
        xs = np.unique(np.append(xs, [np.nextafter(0., -np.inf), 0.]))
        curve = exponential_membership(xs, lam)
        ax.plot(xs, curve, color=TEAL, lw=2, label=f'Спад λ = {lam:g}')
        ax.axhline(alpha, color=ORANGE, ls='--', lw=1.3, label=f'Порог α = {alpha:.4f}')
        ax.fill_between(xs, alpha, curve, where=curve >= alpha, color=TEAL, alpha=.12)
        centers = {}
        for name, number, mu, color in zip('ABC', numbers, membership, CLUSTER_COLORS):
            centers.setdefault((number.center, float(mu)), []).append((name, color))
        previous = []
        for (center, mu), labels in sorted(centers.items()):
            color = labels[0][1] if len(labels) == 1 else INK
            nearby = sum(abs(center - x) < .09 * (high - low) and abs(mu - y) < .10
                         for x, y in previous)
            offset = 8 + 15 * nearby
            ax.scatter(center, mu, color=color, edgecolor=PLOT, s=54, zorder=3)
            ax.annotate(' / '.join(name for name, _ in labels), (center, mu),
                        xytext=(6, -offset if mu > .83 else offset), textcoords='offset points',
                        color=INK, fontsize=9)
            previous.append((center, mu))
        ax.set_title('Принадлежность центров', color=INK, loc='left', fontsize=12)
        ax.set_ylabel('μ(x)')
        ax.set_ylim(-.04, 1.08)
        for item in ax.legend(loc='upper right', fontsize=9, frameon=False).get_texts():
            item.set_color(INK)
        for index, (value, color) in enumerate(zip(numbers, (TEAL, PURPLE, ORANGE))):
            intervals.plot([value.left, value.right], [index, index], color=color, lw=2, marker='o')
            position = (value.center - low) / (high - low)
            alignment = 'left' if position < .12 else 'right' if position > .88 else 'center'
            bounds = f'[{value.left:g}; {value.right:g}]'.replace('-', '−')
            intervals.annotate(bounds, (value.center, index), xytext=(0, 6),
                               textcoords='offset points', ha=alignment, va='bottom',
                               color=color, fontsize=9)
        intervals.set_yticks(range(len(numbers)), ('A', 'B', 'C · результат')[:len(numbers)])
        intervals.set_ylim(len(numbers) - .5, -.65)
        intervals.set_xlim(low - pad, high + pad)
        intervals.set_title('Границы интервалов', color=INK, loc='left', fontsize=11)
        intervals.set_xlabel('Значение x')
        self.draw_idle()

    def plot_clusters(self, samples, mask, metric, full=None, filtered=None):
        widths = np.array([number.right - number.left for number in samples])
        centers = np.array([number.center for number in samples])
        ylabel = {1: 'Ширина интервала', 2: 'Квадрат ширины'}.get(metric, 'Расстояние до медоида')
        for ax, indices, result, title in (
            (self.axes[0], np.ones(len(samples), dtype=bool), full, 'Весь набор'),
            (self.axes[1], mask, filtered, 'После α'),
        ):
            ax.clear()
            self.style_axis(ax)
            ax.set_title(f'{title}  ·  {int(indices.sum())}', color=INK, loc='left', fontsize=11)
            ax.set_xlabel('Центр интервала')
            if ax is self.axes[0]:
                ax.set_ylabel(ylabel if result is not None else 'Ширина интервала')
            if not np.any(indices):
                message = 'Нет интервалов после фильтрации' if samples else 'Сгенерируйте интервалы'
                ax.text(.5, .5, message, transform=ax.transAxes, ha='center', color=MUTED)
                continue
            if result is None and full is not None:
                ax.text(.5, .5, 'Запустите кластеризацию', transform=ax.transAxes,
                        ha='center', color=MUTED, fontsize=9)
                continue
            if result is None:
                ax.scatter(centers[indices], widths[indices], color=TEAL, s=23)
                continue
            labels, distance_to_medoid = result
            ordinates = widths[indices] if metric == 1 else distance_to_medoid
            if metric == 2:
                ordinates = widths[indices]**2
            ax.scatter(centers[indices], ordinates, c=labels % len(CLUSTER_COLORS),
                       cmap=ListedColormap(CLUSTER_COLORS), vmin=0, vmax=len(CLUSTER_COLORS) - 1, s=22)
        if samples:
            first, second = self.axes
            second.set_xlim(first.get_xlim())
            if filtered is not None or full is None:
                lower, upper = zip(first.get_ylim(), second.get_ylim())
                if filtered is not None or mask.any():
                    first.set_ylim(min(lower), max(upper))
                second.set_ylim(first.get_ylim())
        self.draw_idle()
