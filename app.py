"""Запуск приложения, навигация, журнал и справка."""

from datetime import datetime
from pathlib import Path
import sys

from PyQt6.QtCore import QSignalBlocker, Qt
from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtWidgets import (
    QApplication, QButtonGroup, QFileDialog, QFrame, QHBoxLayout, QMainWindow,
    QMessageBox, QPlainTextEdit, QSlider, QStackedWidget, QVBoxLayout, QWidget,
)

from fuzzy_numbers import ALPHA
from ui.clusters import ClustersPage
from ui.storage import Journal
from ui.values import ValuesPage
from ui.widgets import button, card, field_layout, float_box, label, make_scroll, vertical


ROOT = Path(__file__).resolve().parent


class FuzzyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Экспоненциальные нечёткие величины')
        self.setMinimumSize(900, 620)
        area = self.screen().availableGeometry()
        self.resize(min(1360, area.width() - 32), min(820, area.height() - 48))
        self.journal = Journal(ROOT)
        self._build_ui()
        self.values_page.show_values()
        self._write_log('Приложение запущено.')

    def _write_log(self, message: str):
        self.journal.info(message)
        self.log_view.appendPlainText(f'[{datetime.now():%H:%M:%S}] {message}')
        self.status_text.setText(message)

    def _build_ui(self):
        root = QWidget()
        root.setObjectName('root')
        self.setCentralWidget(root)
        shell = vertical(0, (0, 0, 0, 0))
        root.setLayout(shell)

        navigation = QFrame()
        navigation.setObjectName('navigation')
        nav_layout = QHBoxLayout(navigation)
        nav_layout.setContentsMargins(22, 9, 22, 9)
        nav_layout.setSpacing(5)
        nav_layout.addWidget(label('Нечёткие величины', 'brand'))
        nav_layout.addSpacing(23)
        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.nav_buttons = []
        for idx, title in enumerate(('Интервалы', 'Кластеризация', 'Журнал', 'О модели')):
            nav = button(title)
            nav.setObjectName('nav')
            nav.setCheckable(True)
            nav.clicked.connect(lambda _checked=False, i=idx: self.open_page(i))
            self.nav_group.addButton(nav, idx)
            self.nav_buttons.append(nav)
            nav_layout.addWidget(nav)
        nav_layout.addStretch()
        shell.addWidget(navigation)

        content = QWidget()
        content.setObjectName('page')
        outer = vertical(12, (22, 14, 22, 10))
        content.setLayout(outer)
        self.page_title = label('', 'title')
        outer.addWidget(self.page_title)
        self.model = QFrame()
        self.model.setObjectName('model')
        controls = QHBoxLayout(self.model)
        controls.setContentsMargins(14, 9, 14, 9)
        controls.setSpacing(16)
        self.rate = float_box(1.0, 0.01, 100)
        self.rate.setFixedWidth(110)
        self.rate.setKeyboardTracking(False)
        self.rate.setToolTip('Чем больше λ, тем быстрее убывает принадлежность при x > 0.')
        controls.addLayout(field_layout('Скорость спада λ', self.rate))

        self.alpha_input = float_box(ALPHA, 0, 1, decimals=4)
        self.alpha_input.setFixedWidth(94)
        self.alpha_input.setSingleStep(.01)
        self.alpha_input.setKeyboardTracking(False)
        controls.addLayout(field_layout('Порог отбора α', self.alpha_input))
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setAccessibleName('Порог отбора α')
        self.alpha_slider.setFixedHeight(33)
        self.alpha_slider.setToolTip('Сравнивается с принадлежностью центра интервала')
        self.alpha_slider.setRange(0, 10000)
        self.alpha_slider.setValue(round(ALPHA * 10000))
        self.alpha_slider.setSingleStep(100)
        self.alpha_slider.setPageStep(1000)
        self.alpha_slider.setTracking(False)
        controls.addWidget(self.alpha_slider, 1, Qt.AlignmentFlag.AlignBottom)
        controls.addWidget(label('μ(x) = exp(−λx)  при x ≥ 0', 'formula'),
                           alignment=Qt.AlignmentFlag.AlignVCenter)
        outer.addWidget(self.model)

        self.pages = QStackedWidget()
        self.values_page = ValuesPage(self._write_log)
        self.clusters_page = ClustersPage(self._write_log, self.journal)
        self.alpha_slider.valueChanged.connect(self._set_filter)
        self.alpha_input.valueChanged.connect(
            lambda alpha: self.alpha_slider.setValue(round(alpha * 10000))
        )
        self.rate.valueChanged.connect(self._set_filter)
        self.clusters_page.busy_changed.connect(self.alpha_slider.setDisabled)
        self.clusters_page.busy_changed.connect(self.rate.setDisabled)
        self.clusters_page.busy_changed.connect(self.alpha_input.setDisabled)
        self.pages.addWidget(self.values_page)
        self.pages.addWidget(self.clusters_page)
        self.pages.addWidget(self._build_journal_page())
        self.pages.addWidget(self._build_info_page())
        outer.addWidget(self.pages, 1)
        bottom = QHBoxLayout()
        self.status_text = label('Готово к работе.', 'hint', True)
        bottom.addWidget(self.status_text, 1)
        outer.addLayout(bottom)
        shell.addWidget(content, 1)
        self.open_page(0)

    def _set_filter(self):
        alpha = self.alpha_slider.value() / 10000
        lam = self.rate.value()
        with QSignalBlocker(self.alpha_input):
            self.alpha_input.setValue(alpha)
        self.values_page.set_filter(alpha, lam)
        self.clusters_page.set_filter(alpha, lam)

    def open_page(self, index: int):
        titles = ('Интервалы и операции', 'Кластеризация', 'Журнал', 'Математическая модель')
        self.pages.setCurrentIndex(index)
        self.nav_buttons[index].setChecked(True)
        self.page_title.setText(titles[index])
        self.model.setVisible(index in (0, 1))

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
        self.log_view.setPlainText(self.journal.recent())
        surface.layout().addWidget(self.log_view, 1)
        surface.layout().addWidget(label(f'Автоматический файл: {self.journal.path}', 'hint', True))
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
            self.journal.copy_to(Path(path))
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
            ('Экспоненциальная модель',
             'При x ≥ 0 функция μ(x) = exp(−λx), при x < 0 она равна нулю. '
             'Это экспоненциальная плотность, нормированная на максимум λ. Чем больше λ, '
             'тем быстрее падает принадлежность. При x = 1/λ она равна e⁻¹ ≈ 0,368.'),
            ('Фильтрация по α',
             'Интервал проходит фильтр, если степень принадлежности его центра не меньше α. '
             'Арифметические операции выполняются над границами интервалов.'),
            ('Операции и сравнение',
             'Сложение/вычитание: арифметика границ интервалов. Умножение/деление: min/max '
             'попарных комбинаций концов интервалов. Деление запрещено, если интервал делителя '
             'содержит ноль. Если интервалы пересекаются или касаются границами, выводится знак ≈. '
             'Для раздельных интервалов определяется, какой из них расположен левее.'),
            ('Кластерный анализ',
             'k-medoids группирует все интервалы и отдельно повторяет расчёт для тех, которые прошли фильтр. '
             'Медоидом становится элемент с минимальной суммой расстояний до остальных в группе.'),
            ('Структура и использование',
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
        thread = self.clusters_page.thread
        if thread is not None and thread.isRunning():
            QMessageBox.information(self, 'Идёт вычисление',
                                    'Завершите кластеризацию перед закрытием приложения.')
            event.ignore()
            return
        self._write_log('Приложение закрыто.')
        self.journal.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setFont(QFont('Segoe UI', 10))
    palette = app.palette()
    for role, color in (
        (QPalette.ColorRole.Window, '#111722'), (QPalette.ColorRole.Base, '#0e1520'),
        (QPalette.ColorRole.Button, '#223046'), (QPalette.ColorRole.Text, '#ecf3fb'),
        (QPalette.ColorRole.WindowText, '#ecf3fb'), (QPalette.ColorRole.ButtonText, '#ecf3fb'),
        (QPalette.ColorRole.Highlight, '#287d75'), (QPalette.ColorRole.HighlightedText, '#ffffff'),
    ):
        palette.setColor(role, QColor(color))
    app.setPalette(palette)
    app.setStyleSheet((ROOT / 'ui' / 'style.qss').read_text(encoding='utf-8'))
    window = FuzzyWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
