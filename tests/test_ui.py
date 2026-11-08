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
        if cls.window.thread is not None and cls.window.thread.isRunning():
            cls.window.thread.quit()
            cls.window.thread.wait(3000)
        cls.window.close()

    def test_values_and_operation(self):
        self.window.operator.setCurrentIndex(0)
        self.window.run_operation()
        self.assertIn('A + B', self.window.operation_status.text())
        self.assertEqual(len(self.window.values_chart.axes), 2)

    def test_alpha_slider_uses_plotted_values(self):
        self.window.a_a.setValue(2)
        self.window.show_values()
        try:
            self.window.a_a.setValue(3)
            self.window.alpha_slider.setValue(100)
            self.assertIn('Aα: [2.000; 2.000]', self.window.alpha_result.text())
            self.window.show_values()
            self.assertIn('Aα: [3.000; 3.000]', self.window.alpha_result.text())
        finally:
            self.window.a_a.setValue(2)
            self.window.alpha_slider.setValue(50)
            self.window.show_values()

    def test_generation_and_clustering(self):
        self.window.n_samples.setValue(100)
        self.window.generate_data()
        self.assertEqual(self.window.table.rowCount(), 100)
        self.window.k_count.setValue(4)
        self.window.cluster_data()
        # Результаты worker->GUI доставляются Qt event loop, не обычным sleep.
        loop = QEventLoop()
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)
        timer.start(8000)
        self.window.thread.finished.connect(loop.quit)
        loop.exec()
        self.assertIsNotNone(self.window.cluster_labels)
        self.assertEqual(len(set(self.window.cluster_labels)), 4)
        self.assertTrue(self.window.cluster_btn.isEnabled())


if __name__ == '__main__':
    unittest.main()
