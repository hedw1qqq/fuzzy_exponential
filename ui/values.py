"""Ввод нечётких величин, вычисления и фильтрация интервалов."""

from collections.abc import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QFrame, QGridLayout, QHBoxLayout, QMessageBox, QSplitter, QWidget

from fuzzy_numbers import ALPHA, FuzzyNumber, calculate, compare, exponential_membership
from .widgets import Chart, button, card, float_box, label, make_scroll, vertical


class ValuesPage(QWidget):
    def __init__(self, write_log: Callable[[str], None]):
        super().__init__()
        self._write_log = write_log
        self.alpha = ALPHA
        self.lam = 1.0
        self.displayed: tuple[FuzzyNumber, ...] = ()
        self.setObjectName('page')
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        split = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(split)

        controls_widget = QWidget()
        controls_widget.setObjectName('page')
        controls = vertical(0, (0, 0, 0, 0))
        controls_widget.setLayout(controls)
        params = card(vertical(7, (16, 12, 16, 12)))
        form = params.layout()
        form.addWidget(label('Операнды', 'section'))
        bounds = QGridLayout()
        bounds.setHorizontalSpacing(8)
        bounds.setVerticalSpacing(6)
        bounds.setColumnStretch(1, 1)
        bounds.setColumnStretch(2, 1)
        bounds.addWidget(label('Левая граница', 'field'), 0, 1)
        bounds.addWidget(label('Правая граница', 'field'), 0, 2)
        self.left_a, self.right_a = self._add_number_fields(bounds, 1, 'A', 2.0, 3.0)
        self.left_b, self.right_b = self._add_number_fields(bounds, 2, 'B', 5.0, 6.0)
        form.addLayout(bounds)
        operation_label = label('Операция', 'field')
        self.operator = QComboBox()
        operation_label.setBuddy(self.operator)
        self.operator.setAccessibleName('Операция')
        self.operator.addItems(['Сложение  A + B', 'Вычитание  A − B', 'Умножение  A × B', 'Деление  A ÷ B'])
        operation = QHBoxLayout()
        operation.addWidget(operation_label)
        operation.addWidget(self.operator, 1)
        form.addLayout(operation)
        operations = QHBoxLayout()
        self.calc_button = button('Вычислить', primary=True)
        self.calc_button.clicked.connect(self.run_operation)
        self.compare_button = button('Сравнить')
        self.compare_button.clicked.connect(self.run_comparison)
        operations.addWidget(self.calc_button)
        operations.addWidget(self.compare_button)
        self.plot_button = button('A и B')
        self.plot_button.setAccessibleName('Показать интервалы A и B')
        self.plot_button.setToolTip('Показать исходные интервалы и скрыть результат')
        self.plot_button.clicked.connect(self.show_values)
        operations.addWidget(self.plot_button)
        form.addLayout(operations)

        divider = QFrame()
        divider.setObjectName('divider')
        divider.setFixedHeight(1)
        form.addWidget(divider)
        form.addWidget(label('Результат', 'section'))
        self.operation_status = label('Выберите операцию или сравнение.', 'result', True)
        form.addWidget(self.operation_status)
        form.addWidget(label('Отбор по центрам', 'section'))
        self.alpha_result = label('', 'readout')
        form.addWidget(self.alpha_result)
        controls.addWidget(params)
        controls.addStretch(1)
        left = make_scroll(controls_widget)
        left.setMinimumWidth(292)
        split.addWidget(left)

        chart_frame = card(vertical(0, (8, 6, 8, 6)))
        chart_layout = chart_frame.layout()
        self.values_chart = Chart()
        chart_layout.addWidget(self.values_chart, 1)
        split.addWidget(chart_frame)
        split.setSizes([320, 980])
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)

    @staticmethod
    def _add_number_fields(grid, row, name, lower, upper):
        left = float_box(lower, -100, 100)
        right = float_box(upper, -100, 100)
        grid.addWidget(label(name, 'operand'), row, 0)
        grid.addWidget(left, row, 1)
        grid.addWidget(right, row, 2)
        left.setAccessibleName(f'{name}, левая граница')
        right.setAccessibleName(f'{name}, правая граница')
        return left, right

    def _read_values(self):
        return (FuzzyNumber(self.left_a.value(), self.right_a.value()),
                FuzzyNumber(self.left_b.value(), self.right_b.value()))

    def set_filter(self, alpha: float, lam: float):
        self.alpha, self.lam = alpha, lam
        if self.displayed:
            self._plot_values(*self.displayed)

    def _plot_values(self, *numbers: FuzzyNumber):
        self.displayed = numbers
        membership = exponential_membership([number.center for number in numbers], self.lam)
        self.values_chart.plot_values(numbers, membership, self.lam, self.alpha)
        self.alpha_result.setText('\n'.join(
            f'{name}    μ = {mu:.4f}    {"прошёл" if mu >= self.alpha else "отсеян"}'
            for name, mu in zip(('A', 'B', 'C'), membership)
        ))

    def show_values(self):
        try:
            first, second = self._read_values()
            self._plot_values(first, second)
            self.operation_status.setText('Интервалы A и B построены.')
            self._write_log(f'A [{first.left:g}; {first.right:g}], B [{second.left:g}; {second.right:g}].')
        except ValueError as exc:
            QMessageBox.warning(self, 'Неверные границы', str(exc))

    def run_operation(self):
        try:
            first, second = self._read_values()
            op = ('+', '-', '*', '/')[self.operator.currentIndex()]
            result = calculate(first, second, op)
            self._plot_values(first, second, result)
            message = f'A {op} B = [{result.left:.5g}; {result.right:.5g}]'
            self.operation_status.setText(message)
            self._write_log(f'[{first.left:g}; {first.right:g}] {op} '
                            f'[{second.left:g}; {second.right:g}] = [{result.left:.5g}; {result.right:.5g}].')
        except (ValueError, OverflowError) as exc:
            QMessageBox.warning(self, 'Операция не выполнена', str(exc))
            self._write_log(f'Ошибка операции: {exc}')

    def run_comparison(self):
        try:
            first, second = self._read_values()
            sign = compare(first, second)
            self._plot_values(first, second)
            detail = 'Интервалы пересекаются.' if sign == '≈' else 'Интервалы расположены раздельно.'
            message = f'A {sign} B. {detail}'
            self.operation_status.setText(message)
            self._write_log(message)
        except ValueError as exc:
            QMessageBox.warning(self, 'Сравнение не выполнено', str(exc))
