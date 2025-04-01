import sys
import cv2
import numpy as np
import json

# Импорты из PyQt5
from PyQt5 import QtWidgets, QtGui, QtCore, uic
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPixmapItem, QTableWidgetItem
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import pyqtSignal, QObject

class SettingsManager:
    def __init__(self, config_file="config.json"):
        self.config_file = config_file
        self.default_config = {
            "pos_card_center": [0, 210, 0],
            "pos_camera_home": [0, 160, 180],
            "xyz_robots_card_down": [-210, 0, 10],
            "xyz_human_card_down": [210, 0, 30],
            "pos_camera_home_start": [-150, 0, 180],
            "pos_camera_home_fuman": [150, 0, 180],
            "z_step": 0.45
        }

    def load_config(self):
        try:
            with open(self.config_file, "r") as f:
                config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            config = self.default_config.copy()
            self.save_config(config)
        return config

    def save_config(self, config):
        with open(self.config_file, "w") as f:
            json.dump(config, f, indent=4)

    def populate_table(self, table_widget: QtWidgets.QTableWidget):
        config = self.load_config()
        table_widget.setRowCount(len(config))
        table_widget.setColumnCount(2)
        table_widget.setHorizontalHeaderLabels(["Параметр", "Значение"])
        for i, (key, value) in enumerate(config.items()):
            key_item = QTableWidgetItem(key)
            value_item = QTableWidgetItem(str(value))
            table_widget.setItem(i, 0, key_item)
            table_widget.setItem(i, 1, value_item)
        table_widget.resizeColumnsToContents()

    def read_table(self, table_widget: QtWidgets.QTableWidget):
        """Считываем значения из таблицы и возвращаем словарь с настройками."""
        config = {}
        for i in range(table_widget.rowCount()):
            key_item = table_widget.item(i, 0)
            value_item = table_widget.item(i, 1)
            if key_item is None or value_item is None:
                continue
            key = key_item.text()
            value_str = value_item.text()
            try:
                # Попытка интерпретировать строку как список или число
                value = json.loads(value_str)
            except json.JSONDecodeError:
                # Если это не JSON, оставляем как строку
                value = value_str
            config[key] = value
        return config

class MainApp(QObject):
    # В PyQt5 используется pyqtSignal вместо Signal из PySide
    start_signal = pyqtSignal()       # Сигнал для начала работы
    stop_signal = pyqtSignal()        # Сигнал для остановки работы
    pause_signal = pyqtSignal(bool)   # Сигнал для паузы/возобновления
    close_signal = pyqtSignal()       # Сигнал для закрытия GUI

    def __init__(self, app, is_online):
        """
        Принимаем QApplication из main-программы (app)
        и флаг is_online — статус подключения к интернету.
        """
        super().__init__()
        self.app = app

        # Загружаем интерфейс из .ui-файла через uic
        # Убедитесь, что "new_gui001.ui" находится в том же каталоге или указывайте полный путь
        self.window = uic.loadUi("new_gui001.ui")

        # Создаём словарь для хранения QGraphicsScene (для отображения картинок)
        self.scenes = {}

        self.is_running = False
        self.is_paused = False

        # Ищем нужные виджеты по objectName
        self.start_button = self.window.findChild(QtWidgets.QPushButton, "ButtonStart")
        self.pause_button = self.window.findChild(QtWidgets.QPushButton, "ButtonPause")
        self.stop_button = self.window.findChild(QtWidgets.QPushButton, "ButtonStop")

        self.play_statusCircle = self.window.findChild(QtWidgets.QLabel, "play_statusCircle")
        if self.play_statusCircle:
            # Задаём начальный цвет индикатора: красный
            self.play_statusCircle.setStyleSheet("background-color: red; border-radius: 20px;")

        # Подключаем слоты к сигналам от кнопок
        if self.start_button:
            self.start_button.clicked.connect(self.start_pressed)
        if self.pause_button:
            self.pause_button.clicked.connect(self.pause_pressed)
        if self.stop_button:
            self.stop_button.clicked.connect(self.stop_pressed)

        self.status_circle = self.window.findChild(QtWidgets.QLabel, "connection_statusCircle")
        if self.status_circle:
            color = "green" if is_online else "red"
            self.status_circle.setStyleSheet(f"background-color: {color}; border-radius: 10px;")

        # Секция "Настройки"
        self.settings_table = self.window.findChild(QtWidgets.QTableWidget, "settingsTableWidget")
        self.save_button = self.window.findChild(QtWidgets.QPushButton, "saveButton")
        self.apply_button = self.window.findChild(QtWidgets.QPushButton, "applyButton")

        # Инициализируем менеджер настроек
        self.settings_manager = SettingsManager("config.json")
        if self.settings_table:
            self.settings_manager.populate_table(self.settings_table)

        if self.save_button:
            self.save_button.clicked.connect(self.save_settings)
        if self.apply_button:
            self.apply_button.clicked.connect(self.apply_settings)

        # Переопределяем обработчик закрытия окна
        # (В PyQt5 это тоже сработает, хотя обычно делают через переопределение метода в классе окна)
        self.window.closeEvent = self.on_close

        # Показываем окно (но не запускаем цикл событий здесь)
        self.window.show()

    def on_close(self, event):
        """Вызывается при закрытии GUI"""
        print("GUI закрывается. Полное завершение программы.")
        self.close_signal.emit()  # Сигнал о закрытии GUI
        event.accept()
        self.app.quit()  # Закрываем QApplication

    def start_pressed(self):
        """Обработчик кнопки Start"""
        if not self.is_running:
            self.is_running = True
            self.start_signal.emit()
            if self.play_statusCircle:
                self.play_statusCircle.setStyleSheet("background-color: green; border-radius: 20px;")

    def pause_pressed(self):
        """Обработчик кнопки Pause"""
        self.is_paused = not self.is_paused
        self.pause_signal.emit(self.is_paused)
        if self.play_statusCircle:
            if self.is_paused:
                self.play_statusCircle.setStyleSheet("background-color: yellow; border-radius: 20px;")
            else:
                self.play_statusCircle.setStyleSheet("background-color: green; border-radius: 20px;")

    def stop_pressed(self):
        """Обработчик кнопки Stop"""
        self.is_running = False
        self.stop_signal.emit()
        if self.play_statusCircle:
            self.play_statusCircle.setStyleSheet("background-color: red; border-radius: 20px;")

    def load_image(self, img_bgr, view_name):
        """Отображает изображение в QGraphicsView"""
        graphics_view = self.window.findChild(QtWidgets.QGraphicsView, view_name)
        if graphics_view is None:
            print(f"Ошибка: Не найден объект {view_name}")
            return

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Подгоняем размер к QGraphicsView
        view_width = graphics_view.width()
        view_height = graphics_view.height()
        img_resized = cv2.resize(img_rgb, (view_width, view_height), interpolation=cv2.INTER_AREA)

        h, w, ch = img_resized.shape
        bytes_per_line = ch * w
        q_image = QImage(img_resized.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        if view_name not in self.scenes:
            self.scenes[view_name] = QGraphicsScene()
        self.scenes[view_name].clear()
        self.scenes[view_name].addItem(QGraphicsPixmapItem(pixmap))
        graphics_view.setScene(self.scenes[view_name])

        # Принудительно обрабатываем события (обновление интерфейса)
        self.app.processEvents()

    def update_lcd_value(self, new_value, obj_name):
        """Обновляет значение на QLCDNumber"""
        lcd_widget = self.window.findChild(QtWidgets.QLCDNumber, obj_name)
        if lcd_widget:
            lcd_widget.display(new_value)
        else:
            print(f"Ошибка: Не найден {obj_name}")

    def save_settings(self):
        """Сохраняет текущие значения из таблицы в файл"""
        if self.settings_table:
            config = self.settings_manager.read_table(self.settings_table)
            self.settings_manager.save_config(config)
            print("Настройки сохранены:", config)

    def apply_settings(self):
        """Подгружает и применяет настройки из файла"""
        config = self.settings_manager.load_config()
        # Переводим их в атрибуты
        self.pos_card_center = tuple(config.get("pos_card_center", [0, 210, 0]))
        self.pos_camera_home = tuple(config.get("pos_camera_home", [0, 160, 180]))
        self.xyz_robots_card_down = tuple(config.get("xyz_robots_card_down", [-210, 0, 10]))
        self.xyz_human_card_down = tuple(config.get("xyz_human_card_down", [210, 0, 30]))
        self.pos_camera_home_start = tuple(config.get("pos_camera_home_start", [-150, 0, 180]))
        self.pos_camera_home_fuman = tuple(config.get("pos_camera_home_fuman", [150, 0, 180]))
        self.z_step = config.get("z_step", 0.45)

        print("Настройки применены:", config)
        # Если нужно, обновляем таблицу
        if self.settings_table:
            self.settings_manager.populate_table(self.settings_table)

# Пример main-функции, если нужно
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    is_online = False
    gui = MainApp(app, is_online)
    sys.exit(app.exec_())
