"""Минимальные smoke-тесты PyQt6 GUI.

В среде без Qt тесты корректно пропускаются. Для headless Linux:
QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests -v
"""
import csv
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fuzzy_numbers import ALPHA

if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

try:
    from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox
    from PyQt6.QtCore import QEventLoop, QTimer
    from app import FuzzyWindow
except ImportError:
    QApplication = None


@unittest.skipIf(QApplication is None, 'PyQt6 отсутствует в тестовом окружении')
class GuiSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        directory = TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        with patch('app.ROOT', Path(directory.name)):
            cls.window = FuzzyWindow()

    @classmethod
    def tearDownClass(cls):
        thread = cls.window.clusters_page.thread
        if thread is not None and thread.isRunning():
            thread.quit()
            thread.wait(3000)
        cls.window.close()

    def test_values_and_operation(self):
        page = self.window.values_page
        page.operator.setCurrentIndex(0)
        page.run_operation()
        self.assertEqual(page.operation_status.text(), 'A + B = [7; 9]')
        membership, intervals = page.values_chart.axes
        self.assertEqual(list(membership.lines[-1].get_ydata()), [ALPHA, ALPHA])
        self.assertEqual(list(intervals.lines[-1].get_xdata()), [7, 9])
        slider = self.window.alpha_slider
        try:
            slider.setValue(0)
            self.assertEqual(list(intervals.lines[-1].get_xdata()), [7, 9])
            self.assertIn('прошёл', page.alpha_result.text())
            self.assertEqual(page.operation_status.text(), 'A + B = [7; 9]')
        finally:
            slider.setValue(round(ALPHA * 10000))

    def test_coincident_centers_share_one_label(self):
        page = self.window.values_page
        fields = (page.left_a, page.right_a, page.left_b, page.right_b)
        saved = [field.value() for field in fields]
        try:
            for field, value in zip(fields, (2, 2, 1, 3)):
                field.setValue(value)
            page.run_operation()
            top, bottom = page.values_chart.axes
            self.assertEqual([marker.get_text() for marker in top.texts], ['A / B', 'C'])
            self.assertEqual(top.get_xlim(), bottom.get_xlim())
            self.assertEqual([item.get_text() for item in bottom.texts], ['[2; 2]', '[1; 3]', '[3; 5]'])
            self.assertEqual(bottom.get_yticklabels()[-1].get_text(), 'C · результат')
        finally:
            for field, value in zip(fields, saved):
                field.setValue(value)
            page.show_values()

    def test_generation_and_clustering(self):
        page = self.window.clusters_page
        page.n_samples.setValue(100)
        page.generate_data()
        self.assertEqual(page.table.rowCount(), 100)
        page.k_count.setValue(4)
        page.cluster_data()
        slider = self.window.alpha_slider
        self.assertFalse(slider.isEnabled())
        loop = QEventLoop()
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)
        timer.start(8000)
        page.thread.finished.connect(loop.quit)
        loop.exec()
        self.assertIsNotNone(page.full)
        self.assertEqual(len(set(page.full[0])), 4)
        self.assertIsNotNone(page.filtered)
        self.assertLess(len(page.filtered[0]), 100)
        self.assertTrue(slider.isEnabled())

        path = self.window.journal.path.with_name('samples.csv')
        with patch.object(QFileDialog, 'getSaveFileName', return_value=(str(path), '')):
            with patch.object(QMessageBox, 'information'):
                page.export_btn.click()
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream, delimiter=';'))
        self.assertEqual(len(rows), 100)
        self.assertEqual({float(row['alpha']) for row in rows}, {ALPHA})
        self.assertEqual(sum(bool(row['filtered_cluster']) for row in rows), len(page.filtered[0]))
        try:
            sample = page.samples[0]
            slider.setValue(10000)
            self.assertIs(page.samples[0], sample)
            self.assertIsNotNone(page.full)
            self.assertIsNone(page.filtered)
            self.assertEqual(page.table.item(0, 5).text(), 'Отсеян')
        finally:
            slider.setValue(round(ALPHA * 10000))


if __name__ == '__main__':
    unittest.main()
