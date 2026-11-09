"""Запуск приложения, навигация, журнал и справка."""

from datetime import datetime
from pathlib import Path
import sys

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QButtonGroup, QFileDialog, QFrame, QHBoxLayout, QMainWindow,
    QMessageBox, QPlainTextEdit, QStackedWidget, QVBoxLayout, QWidget,
)

from ui.clusters import ClustersPage
from ui.storage import Journal
from ui.values import ValuesPage
from ui.widgets import button, card, label, make_scroll, vertical


ROOT = Path(__file__).resolve().parent


class FuzzyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Экспоненциальные нечёткие величины')
        self.resize(1380, 860)
        self.setMinimumSize(950, 660)
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
        self.values_page = ValuesPage(self._write_log)
        self.clusters_page = ClustersPage(self._write_log, self.journal)
        self.pages.addWidget(self.values_page)
        self.pages.addWidget(self.clusters_page)
        self.pages.addWidget(self._build_journal_page())
        self.pages.addWidget(self._build_info_page())
        outer.addWidget(self.pages, 1)
        bottom = QHBoxLayout()
        self.status_text = label('Готово к работе.', 'hint', True)
        bottom.addWidget(self.status_text, 1)
        bottom.addWidget(label('Aα = [a, a − ln(α)/λ]', 'hint'))
        outer.addLayout(bottom)
        shell.addWidget(content, 1)
        self.open_page(0)

    def open_page(self, index: int):
        titles = [
            ('Величины и операции', 'Один α-срез для каждой величины, интервальная арифметика'),
            ('Кластеризация', '100–300 α-интервалов: группировка по k или порогу d'),
            ('Журнал', 'История действий программы с сохранением в файл'),
            ('О математической модели', 'Формулы, принятые соглашения и ограничения'),
        ]
        self.pages.setCurrentIndex(index)
        self.nav_buttons[index].setChecked(True)
        self.page_title.setText(titles[index][0])
        self.page_subtitle.setText(titles[index][1])

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
            ('01  Экспоненциальная модель',
             'Исходная функция: μ(x)=0 при x<a, exp(−λ(x−a)) при x≥a, λ>0. '
             'Это плотность экспоненциального распределения, нормированная на максимум.'),
            ('02  Один α-срез = один интервал',
             'Для выбранного α (0<α≤1) величина представляется одним отрезком '
             'Aα=[a, a−ln(α)/λ]. Два параметра a и λ позволяют показать исходную '
             'кривую, но для операций используется только один интервал на величину.'),
            ('03  Операции и сравнение',
             'Арифметика интервалов при одном общем α: сложение и вычитание концов, '
             'для умножения/деления минимум и максимум четырёх комбинаций. '
             'Если делитель содержит ноль, деление не допускается. '
             'Сравнение выполняется по серединам интервалов.'),
            ('04  Кластеризация',
             'Генерируется 100–300 исходных экспоненциальных величин. '
             'Для каждой вычисляется один α-срез на общем уровне. Расстояние '
             'между интервалами — RMS-различие двух границ. Complete linkage '
             'останавливается по числу кластеров k или порогу d. '
             'График показывает положение и размах отрезков.'),
            ('05  Ограничение представления',
             'Отдельный α-срез не определяет в общем случае всю нечёткую величину. '
             'Результаты операций — интервалы на выбранном уровне, а не '
             'полные функции принадлежности. Для другого α расчёт повторяется. '
             'Подробности — в README.md.'),
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
    app.setStyleSheet((ROOT / 'ui' / 'style.qss').read_text(encoding='utf-8'))
    window = FuzzyWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
