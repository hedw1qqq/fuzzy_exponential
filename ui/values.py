"""Ввод нечётких величин, вычисления и просмотр α-срезов."""

from collections.abc import Callable

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QFrame, QHBoxLayout, QMessageBox, QSlider, QSplitter, QWidget

from fuzzy_numbers import FuzzyNumber, calculate, compare, exponential
from .widgets import Chart, button, card, field_layout, float_box, label, make_scroll, vertical


class ValuesPage(QWidget):
    def __init__(self, write_log: Callable[[str], None]):
        super().__init__()
        self._write_log = write_log
        self.plotted_numbers: list[FuzzyNumber] = []
        self._last_action = 'values'
        self._last_operator = '+'
        self.setObjectName('page')
        layout = QHBoxLayout(self)
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
        self.a_a, self.l_a = self._add_number_fields(form, 'Величина A', 2.0, 1.1)
        self.a_b, self.l_b = self._add_number_fields(form, 'Величина B', 5.0, 1.7)
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
        cuts.addWidget(label('Один α-срез на величину', 'section'))
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

    @staticmethod
    def _add_number_fields(form, title, a, lam):
        form.addWidget(label(title, 'section'))
        shift = float_box(a, -100, 100)
        rate = float_box(lam, 0.01, 100)
        form.addLayout(field_layout('Сдвиг a', shift))
        form.addLayout(field_layout('Интенсивность λ (> 0)', rate))
        return shift, rate

    def _read_values(self):
        a, la = self.a_a.value(), self.l_a.value()
        b, lb = self.a_b.value(), self.l_b.value()
        level = self.alpha_slider.value() / 100
        A = exponential(a, la, 'A', alpha=level)
        B = exponential(b, lb, 'B', alpha=level)
        return a, la, b, lb, A, B

    def _plot_values(self, a, la, b, lb, A: FuzzyNumber, B: FuzzyNumber,
                     result: FuzzyNumber | None = None):
        self.plotted_numbers = [A, B] + ([result] if result is not None else [])
        self.values_chart.plot_values(a, la, b, lb, A, B, A.alpha, result)
        self.update_alpha_cuts(redraw=False)

    def update_alpha_cuts(self, _value: int | None = None, *, redraw: bool = True):
        level = self.alpha_slider.value() / 100
        self.alpha_label.setText(f'α = {level:.2f}')
        if redraw:
            # Меняется α -> пересчитываются границы интервалов и результат операции.
            self._refresh(log=False)
        else:
            self.alpha_result.setText('\n'.join(
                f'{name}α: [{number.left[0]:.3f}; {number.right[0]:.3f}]'
                for name, number in zip(('A', 'B', 'C'), self.plotted_numbers)
            ))

    def _refresh(self, *, log: bool = True):
        try:
            a, la, b, lb, A, B = self._read_values()
            level = A.alpha
            if self._last_action == 'operation':
                op = self._last_operator
                C = calculate(A, B, op)
                self._plot_values(a, la, b, lb, A, B, C)
                self.operation_status.setText(
                    f'A {op} B на α={level:.2f}\n'
                    f'Результат: [{C.left[0]:.3f}; {C.right[0]:.3f}]\n'
                    f'Середина интервала: {C.representative:.3f}'
                )
                if log:
                    self._write_log(
                        f'A(a={a:g}, λ={la:g}) {op} B(a={b:g}, λ={lb:g}), '
                        f'α={level:.2f}: [{C.left[0]:.5g}; {C.right[0]:.5g}].')
            elif self._last_action == 'comparison':
                sign = compare(A, B)
                self._plot_values(a, la, b, lb, A, B)
                self.operation_status.setText(
                    f'A {sign} B на α={level:.2f}\n'
                    f'Середина A: {A.representative:.3f}\n'
                    f'Середина B: {B.representative:.3f}\n'
                    'Сравнение середин двух интервалов.')
                if log:
                    self._write_log(
                        f'Сравнение A {sign} B при α={level:.2f}: '
                        f'{A.representative:.5g} и {B.representative:.5g}.')
            else:
                self._plot_values(a, la, b, lb, A, B)
                self.operation_status.setText(
                    f'α={level:.2f}\n'
                    f'A: [{A.left[0]:.3f}; {A.right[0]:.3f}]\n'
                    f'B: [{B.left[0]:.3f}; {B.right[0]:.3f}]')
                if log:
                    self._write_log(
                        f'Построены A(a={a:g}, λ={la:g}) и B(a={b:g}, λ={lb:g}), '
                        f'α={level:.2f}.')
        except (ValueError, OverflowError) as exc:
            self.operation_status.setText(f'Ошибка: {exc}')
            if log:
                QMessageBox.warning(self, 'Невозможно выполнить действие', str(exc))
                self._write_log(f'Ошибка операции: {exc}')

    def show_values(self):
        self._last_action = 'values'
        self._refresh()

    def run_operation(self):
        self._last_action = 'operation'
        self._last_operator = ('+', '-', '*', '/')[self.operator.currentIndex()]
        self._refresh()

    def run_comparison(self):
        self._last_action = 'comparison'
        self._refresh()
