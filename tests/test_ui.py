"""Минимальные smoke-тесты PyQt6 GUI.

В среде без Qt тесты корректно пропускаются. Для headless Linux:
QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests -v
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
        self.assertIn('A + B', page.operation_status.text())
        self.assertEqual(len(page.values_chart.axes), 2)

    def test_alpha_slider_uses_plotted_values(self):
        page = self.window.values_page
        page.a_a.setValue(2)
        page.show_values()
        try:
            marker = page.values_chart.alpha_line
            page.a_a.setValue(3)
            page.alpha_slider.setValue(100)
            self.assertIs(page.values_chart.alpha_line, marker)
            self.assertEqual(list(marker.get_ydata()), [1.0, 1.0])
            self.assertIn('Aα: [2.000; 2.000]', page.alpha_result.text())
            page.show_values()
            self.assertIn('Aα: [3.000; 3.000]', page.alpha_result.text())
        finally:
            page.a_a.setValue(2)
            page.alpha_slider.setValue(50)
            page.show_values()

    def test_generation_and_clustering(self):
        page = self.window.clusters_page
        page.n_samples.setValue(100)
        page.generate_data()
        self.assertEqual(page.table.rowCount(), 100)
        page.k_count.setValue(4)
        page.cluster_data()
        # Результаты worker->GUI доставляются Qt event loop, не обычным sleep.
        loop = QEventLoop()
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)
        timer.start(8000)
        page.thread.finished.connect(loop.quit)
        loop.exec()
        self.assertIsNotNone(page.cluster_labels)
        self.assertEqual(len(set(page.cluster_labels)), 4)
        self.assertTrue(page.cluster_btn.isEnabled())


if __name__ == '__main__':
    unittest.main()
