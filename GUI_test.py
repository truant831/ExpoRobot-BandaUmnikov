import sys
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtUiTools import QUiLoader
from PyQt5.QtCore import QFile, QIODevice, QTimer

class MainApp:
    def __init__(self):
        # Создание приложения
        self.app = QApplication(sys.argv)

        # Загрузка UI-файла
        ui_file_name = "new_gui001.ui"  # Убедись, что путь корректный
        ui_file = QFile(ui_file_name)
        if not ui_file.open(QIODevice.ReadOnly):
            print(f"Cannot open {ui_file_name}: {ui_file.errorString()}")
            sys.exit(-1)
        
        loader = QUiLoader()
        self.window = loader.load(ui_file)
        ui_file.close()
        
        if not self.window:
            print(loader.errorString())
            sys.exit(-1)

        # Найти объекты интерфейса
        self.graphics_view = self.window.findChild(QtWidgets.QGraphicsView, "graphicsViewCamera")
        self.lcd_number = self.window.findChild(QtWidgets.QLCDNumber, "lcdNumberRobot")

        # Проверка, нашли ли объекты
        if not self.graphics_view or not self.lcd_number:
            print("Ошибка: Не удалось найти widgets в UI-файле")
            sys.exit(-1)

        # Создаем словарь для хранения сцен QGraphicsScene
        self.scenes = {}

        # Устанавливаем начальное значение LCD
        self.robot_value = 0
        self.lcd_number.display(self.robot_value)

        # Загружаем изображение (пример: берём тестовое изображение)
        test_image = cv2.imread("Pictures/test_img_camera.jpg")  # Укажи путь к изображению
        if test_image is not None:
            self.load_image(test_image, "graphicsViewCamera")

        # Таймер для обновления LCD
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_lcd_value)
        self.timer.start(1000)  # Обновление каждую секунду

        self.window.show()
        sys.exit(self.app.exec())

    def load_image(self, img_bgr, view_name):
        """Универсальная функция загрузки и отображения изображения в QGraphicsView.

        Args:
            img_bgr (numpy.ndarray): Изображение в формате OpenCV (BGR).
            view_name (str): Имя объекта QGraphicsView в UI, где нужно отобразить изображение.
        """
        # Найти QGraphicsView по имени
        graphics_view = self.window.findChild(QtWidgets.QGraphicsView, view_name)
        if graphics_view is None:
            print(f"Ошибка: Не найден объект {view_name}")
            return
        
        # Преобразование BGR → RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Получение размеров виджета
        view_width = graphics_view.width()
        view_height = graphics_view.height()

        # Масштабирование изображения под размер объекта
        img_resized = cv2.resize(img_rgb, (view_width, view_height), interpolation=cv2.INTER_AREA)

        # Преобразование в QImage
        h, w, ch = img_resized.shape
        bytes_per_line = ch * w
        q_image = QImage(img_resized.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        # Отображение в QGraphicsView
        if view_name not in self.scenes:
            self.scenes[view_name] = QGraphicsScene()
        self.scenes[view_name].clear()
        self.scenes[view_name].addItem(QGraphicsPixmapItem(pixmap))
        graphics_view.setScene(self.scenes[view_name])

    def update_lcd_value(self):
        """Обновляет значение на LCD"""
        self.robot_value += 1  # Просто увеличиваем на 1 для примера
        self.lcd_number.display(self.robot_value)

if __name__ == "__main__":
    MainApp()
