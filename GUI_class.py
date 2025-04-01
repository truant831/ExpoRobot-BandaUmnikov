import sys
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap, QImage
from PySide6.QtUiTools import QUiLoader
from PyQt5.QtCore import QFile, QIODevice, Signal, QObject

import json

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
            key_item = QtWidgets.QTableWidgetItem(key)
            # Преобразуем список или число в строку
            value_item = QtWidgets.QTableWidgetItem(str(value))
            table_widget.setItem(i, 0, key_item)
            table_widget.setItem(i, 1, value_item)
        table_widget.resizeColumnsToContents()

    def read_table(self, table_widget: QtWidgets.QTableWidget):
        """Считываем значения из таблицы и возвращаем конфигурацию"""
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
                value = value_str
            config[key] = value
        return config


class MainApp(QObject):
    start_signal = Signal()  # Сигнал для начала работы
    stop_signal = Signal()  # Сигнал для остановки работы
    pause_signal = Signal(bool)  # Сигнал для паузы/возобновления работы
    close_signal = Signal()  # Сигнал для закрытия GUI

    def __init__(self, app, is_online):
        """Принимаем QApplication из main-программы и статус подключения к интернету"""
        super().__init__()
        self.app = app  

        # Загрузка UI-файла
        ui_file_name = "new_gui001.ui"  
        ui_file = QFile(ui_file_name)
        if not ui_file.open(QIODevice.ReadOnly):
            print(f"Cannot open {ui_file_name}: {ui_file.errorString()}")
            return
        
        loader = QUiLoader()
        self.window = loader.load(ui_file)
        ui_file.close()
        
        if not self.window:
            print(loader.errorString())
            return

        # Создаем словарь для хранения сцен QGraphicsScene
        self.scenes = {}

        # Устанавливаем начальное состояние кнопок
        self.is_running = False
        self.is_paused = False

        # Находим кнопки в UI
        self.start_button = self.window.findChild(QtWidgets.QPushButton, "ButtonStart")
        self.pause_button = self.window.findChild(QtWidgets.QPushButton, "ButtonPause")
        self.stop_button = self.window.findChild(QtWidgets.QPushButton, "ButtonStop")

        # Для индикатора состояния вместо радио-кнопки используем QLabel
        self.play_statusCircle = self.window.findChild(QtWidgets.QLabel, "play_statusCircle")

        # Устанавливаем начальный цвет индикатора (красный)
        if self.play_statusCircle:
            # Размер 40x40, радиус 20 пикселей для получения круга
            self.play_statusCircle.setStyleSheet("background-color: red; border-radius: 20px;")

        # Подключаем кнопки к методам
        if self.start_button:
            self.start_button.clicked.connect(self.start_pressed)
        if self.pause_button:
            self.pause_button.clicked.connect(self.pause_pressed)
        if self.stop_button:
            self.stop_button.clicked.connect(self.stop_pressed)

        self.status_circle = self.window.findChild(QtWidgets.QLabel, "connection_statusCircle")
        if self.status_circle:
            # Определяем цвет: зеленый, если is_online True, иначе красный.
            color = "green" if is_online else "red"
            # Применяем стили: закругляем углы, чтобы получить круг.
            self.status_circle.setStyleSheet(f"background-color: {color}; border-radius: 10px;")
        
        # вкладке "Настройки" добавлены:
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

        # Перехватываем событие закрытия окна
        self.window.closeEvent = self.on_close

        # Отображаем окно (но НЕ выходим в цикл событий!)
        self.window.show()

    def on_close(self, event):
        """Вызывается при закрытии GUI"""
        print("GUI закрывается. Полное завершение программы.")
        self.close_signal.emit()  # Отправляем сигнал о закрытии GUI
        event.accept()
        self.app.quit()  # Корректно завершаем цикл событий Qt

    def start_pressed(self):
        """Обработчик кнопки Start"""
        if not self.is_running:
            self.is_running = True
            self.start_signal.emit()  # Отправляем сигнал о начале работы
            if self.play_statusCircle:
                    self.play_statusCircle.setStyleSheet("background-color: green; border-radius: 20px;")

    def pause_pressed(self):
        """Обработчик кнопки Pause"""
        self.is_paused = not self.is_paused
        self.pause_signal.emit(self.is_paused)  # Отправляем сигнал о паузе
         # Если в паузе, цвет меняем на желтый, иначе возвращаем зеленый (если работа продолжается)
        if self.play_statusCircle:
            if self.is_paused:
                self.play_statusCircle.setStyleSheet("background-color: yellow; border-radius: 20px;")
            else:
                self.play_statusCircle.setStyleSheet("background-color: green; border-radius: 20px;")


    def stop_pressed(self):
        """Обработчик кнопки Stop"""
        self.is_running = False
        self.stop_signal.emit()  # Отправляем сигнал об остановке
        # При остановке цвет переводим в красный
        if self.play_statusCircle:
            self.play_statusCircle.setStyleSheet("background-color: red; border-radius: 20px;")

    def load_image(self, img_bgr, view_name):
        """Отображает изображение в QGraphicsView"""
        graphics_view = self.window.findChild(QtWidgets.QGraphicsView, view_name)
        if graphics_view is None:
            print(f"Ошибка: Не найден объект {view_name}")
            return
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

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

        # Принудительно обновляем интерфейс
        self.app.processEvents()

    def update_lcd_value(self, new_value, obj_name):
        """Обновляет значение на LCD"""
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
        """Обновляет переменные из файла настроек и, при необходимости, в основной логике"""
        config = self.settings_manager.load_config()
        # Обновляем внутренние переменные
        self.pos_card_center = tuple(config.get("pos_card_center", [0, 210, 0]))
        self.pos_camera_home = tuple(config.get("pos_camera_home", [0, 160, 180]))
        # Аналогично для остальных переменных:
        self.xyz_robots_card_down = tuple(config.get("xyz_robots_card_down", [-210, 0, 10]))
        self.xyz_human_card_down = tuple(config.get("xyz_human_card_down", [210, 0, 30]))
        self.pos_camera_home_start = tuple(config.get("pos_camera_home_start", [-150, 0, 180]))
        self.pos_camera_home_fuman = tuple(config.get("pos_camera_home_fuman", [150, 0, 180]))
        self.z_step = config.get("z_step", 0.45)
        print("Настройки применены:", config)
        # При необходимости можно обновить и отображение в таблице
        if self.settings_table:
            self.settings_manager.populate_table(self.settings_table)
