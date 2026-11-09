"""Smoke-тесты интерфейса на PyQt6.

Для headless Linux: QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests -v
"""
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

try:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QEventLoop, QTimer
except ImportError:
    QApplication = None

if QApplication is not None:
    from app import FuzzyWindow


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
        page.alpha_slider.setValue(50)
        page.operator.setCurrentIndex(0)
        page.run_operation()
        self.assertIn('A + B', page.operation_status.text())
        self.assertEqual(len(page.values_chart.axes), 2)
        self.assertEqual(len(page.plotted_numbers), 3)
        self.assertTrue(all(len(v.alphas) == 1 for v in page.plotted_numbers))

    def test_changing_alpha_recomputes_intervals(self):
        page = self.window.values_page
        page.a_a.setValue(2)
        page.alpha_slider.setValue(50)
        page.show_values()
        half_right = page.plotted_numbers[0].right[0]
        try:
            page.alpha_slider.setValue(100)
            self.assertEqual(page.plotted_numbers[0].alpha, 1.0)
            self.assertLess(page.plotted_numbers[0].right[0], half_right)
            self.assertIn('Aα: [2.000; 2.000]', page.alpha_result.text())
            page.a_a.setValue(3)
            page.show_values()
            self.assertIn('Aα: [3.000; 3.000]', page.alpha_result.text())
        finally:
            page.a_a.setValue(2)
            page.alpha_slider.setValue(50)
            page.show_values()

    def test_generation_and_clustering(self):
        page = self.window.clusters_page
        page.n_samples.setValue(100)
        page.alpha.setValue(0.4)
        page.generate_data()
        self.assertEqual(page.table.rowCount(), 100)
        self.assertEqual(page.table.columnCount(), 7)
        self.assertTrue(all(s.number.alpha == 0.4 for s in page.samples))
        page.alpha.setValue(0.6)
        self.assertEqual(page.table.rowCount(), 0)
        page.generate_data()
        self.assertTrue(all(s.number.alpha == 0.6 for s in page.samples))
        page.k_count.setValue(4)
        page.cluster_data()
        # Не подключаемся к уже завершённому потоку без проверки.
        thread = page.thread
        if thread is not None:
            loop = QEventLoop()
            timer = QTimer()
            timer.setSingleShot(True)
            timer.timeout.connect(loop.quit)
            thread.finished.connect(loop.quit)
            timer.start(8000)
            loop.exec()
        self.assertIsNotNone(page.cluster_labels)
        self.assertEqual(len(set(page.cluster_labels)), 4)
        self.assertTrue(page.cluster_btn.isEnabled())


if __name__ == '__main__':
    unittest.main()
