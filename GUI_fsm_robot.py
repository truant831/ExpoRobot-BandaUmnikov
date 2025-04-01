import sys
import time
import cv2
import json
import numpy as np
from ast import literal_eval

import os 
directory="/home/jetson/Documents/VSOSH_UTS/"
os.chdir(directory)

filename = "digit_data.json"

from transitions import Machine, State
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread, QObject, pyqtSignal, QTimer, QCoreApplication
from GUI_class_Qt5 import MainApp

from libs.video_predict_frukto2 import (
    predict_card,
    compare_cards_fruito10_min_usage,
    draw_matching_boxes,
    get_card_key,
    extract_and_concatenate,
    find_play_zone_dynamic
)
from libs.YaSpeech import generate_speech, recognize_speech
from libs.arm_control_XYZ import set_position
from libs.serial_pico import RPiPico_serial
from libs.play_sounds import Say_object_class, Say_phraze, Say_object_color
from libs.button_callback import clean_btn, wait_time_or_btn
from libs.video import camera
pico = RPiPico_serial('/dev/ttyTHS1')

import http.client as httplib

def checkInternetHttplib(url="tts.api.cloud.yandex.net", timeout=3):
    connection = httplib.HTTPConnection(url, timeout=timeout)
    try:
        print("Checking Internet connection")
        # only header requested for fast operation
        connection.request("HEAD", "/")
        connection.close()  # connection closed
        print("Internet On")
        return True
    except Exception as exep:
        print(exep)
        return False



# ------------ рабочие потоки ------------
class BeginWorker(QObject):
    newFrameCamera = pyqtSignal(object)
    newFrameZone = pyqtSignal(object)

    requestSlider1Value = pyqtSignal()
    requestSlider2Value = pyqtSignal()
    # newMessage1 = pyqtSignal(str)
    # newMessage2 = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._running = True
        # Значения по умолчанию, чтобы не было ошибки,
        # если сигналы от GUI ещё не пришли:
        self.slider1_value = (0, 255, 0, 255, 0, 255)
        self.slider2_value = (0, 255, 0, 255, 0, 255)

    def stop(self):
        self._running = False
        print("[BeginWorker] Завершен")

    def run(self):
        print("[BeginWorker] Цикл BEGIN")
        try:
            while self._running:
                frame = stream.read()
                # frame = cv2.imread("Pictures/Screenshot from 2025-02-16 15-33-40.png")
                time.sleep(0.02)
                # Запросим значения ВСЕХ слайдеров у GUI
                self.requestSlider1Value.emit()
                self.requestSlider2Value.emit()

                # Отдадим картинку с "камеры" в GUI
                self.newFrameCamera.emit(frame)
                
                # HSV-порог. Пример использования слайдеров.
                HSV_limit1 = self.slider1_value
                HSV_limit2 = self.slider2_value

                zone_img, zone_box = find_play_zone_dynamic(HSV_limit1, HSV_limit2, frame)
                if zone_img is not None:
                    self.newFrameZone.emit(zone_img) 
        except Exception as e:
            print("Ошибка в BeginWorker.run():", e)
    
    def on_slider1_value_received(self, val):
        self.slider1_value = val

    def on_slider2_value_received(self, val):
        self.slider2_value = val


class PlayWorker(QObject):
    # global stream
    newFrameCamera = pyqtSignal(object)

    newFrameMyCard = pyqtSignal(object)
    newFramePlayCard = pyqtSignal(object)
    newFrameLastComboRobot = pyqtSignal(object)
    newFrameLastComboField = pyqtSignal(object)

    scoreUpdated = pyqtSignal(int, int)  # (score_robot, score_human)

    def __init__(self, config):
        super().__init__()
        self._running = True
        self.isSpeaking = False
        self.match_found = False
        # Конфигурация
        self.config = config

        # Извлекаем необходимые переменные из конфигурации
        self.pos_card_center      = config.get("pos_card_center", [0, 210, 0])
        self.pos_camera_home      = config.get("pos_camera_home", [0, 160, 180])
        self.xyz_robots_card_down = config.get("xyz_robots_card_down", [-210, 0, 10])
        self.xyz_human_card_down  = config.get("xyz_human_card_down", [210, 0, 30])
        self.pos_camera_home_start= config.get("pos_camera_home_start", [-150, 0, 180])
        self.pos_camera_home_fuman= config.get("pos_camera_home_fuman", [150, 0, 180])
        self.z_step               = config.get("z_step", 0.45)

        print("[PlayWorker] Загруженные настройки:")
        print("pos_card_center:", self.pos_card_center)
        print("pos_camera_home:", self.pos_camera_home)
        print("xyz_robots_card_down:", self.xyz_robots_card_down)
        print("xyz_human_card_down:", self.xyz_human_card_down)
        print("pos_camera_home_start:", self.pos_camera_home_start)
        print("pos_camera_home_fuman:", self.pos_camera_home_fuman)
        print("z_step:", self.z_step)

        # Счёт
        self.score_robot=0
        self.score_human=0

    def stop(self):
        self._running = False

    def move_card(self, position, adress):
        global match_found, was_clicked
        set_position(tuple(np.add(position, (0,0,20))))
        time.sleep(0.6)
        set_position(position)
        time.sleep(0.8)
        pico.apply(1,'RED')
        time.sleep(1)
        pico.apply(1,'PURPLE')
        set_position(tuple(np.add(position, (0,0,20))))
        time.sleep(0.5)
        if adress=="robot":
            put_position=self.xyz_robots_card_down
        else:
            put_position=self.xyz_human_card_down
        set_position(tuple(np.add(put_position, (0,0,60))))
        time.sleep(0.5)
        match_found=False
        was_clicked=False #Обнуляем нажатия человека, так как карту уже отнесли на полпути текущую, а новую он не мог успеть увидеть
        time.sleep(1.0)
        set_position(put_position)
        time.sleep(0.5)
        pico.apply(0,'BLACK')

    def run(self):
        print("[PlayWorker] Основной цикл PLAY (после init-блока).")
        global img_my_card, my_card_results, my_card_key, img_play_card
        global was_clicked
        global isSpeaking

        time_last_predict = time.time()
        n_detected = 0
        pos_x = 0
        pos_y = 0
        pos_z = 0
        t_pause = 0

        #попробуем найти карту и распознать ее, надо на случай если человек нажмет раньше робота первую карту в колоде, чтобы робот знал где брать ему карту
        frame = stream.read()
        img_card_detected, play_card_results, play_card_key, card_box, zone_box = predict_card(frame)

        (x,y,w,h)=card_box
        card_x=int(x+w/2)
        card_y=int(y+h/2)
        print("Card center " +str(card_x)+" , "+str(card_y))
        was_clicked=False

        while self._running:
            frame = stream.read()
            self.newFrameCamera.emit(frame)
    
            if was_clicked and card_x>0 and card_y>0:
                # Перенос карты фумену
                self.score_human += 1
                self.scoreUpdated.emit(self.score_robot, self.score_human)

                pos_x, pos_y, pos_z = self.pos_card_center
                pos_x = int(0 + (card_x-340)/4)
                pos_y = int(210 - (card_y-250)/4)
                pos_z = pos_z - self.z_step
                self.pos_card_center = (pos_x, pos_y, pos_z)
                was_clicked = False
                self.move_card(self.pos_card_center,"human")
                self.pos_human_card_down=tuple(np.add(self.xyz_human_card_down, (0,0,self.step_z)))
                set_position(self.pos_camera_home)
                time.sleep(1)
            else:
                self.match_found = False
                was_clicked = False

                # Запускаем распознавание не чаще, чем раз в 0.5 + t_pause
                if (time.time()-time_last_predict) > (0.5 + t_pause):
                    img_card_detected, play_card_results, play_card_key, card_box, zone_box = predict_card(frame)
                    time_last_predict = time.time()
                    t_pause = 0
                else:
                    img_card_detected = None

                if img_card_detected is not None:
                    n_detected = len(play_card_results)
                    (x, y, w, h) = card_box
                    card_x = int(x + w/2)
                    card_y = int(y + h/2)
                    print("Card center {}, {}".format(card_x, card_y))

                    pos_x, pos_y, pos_z = self.pos_card_center
                    pos_x = int(0 + (card_x-340)/4)
                    pos_y = int(210 - (card_y-250)/4)

                    self.newFramePlayCard.emit(img_card_detected)
                    time.sleep(4.5)

                    # Сравниваем обнаруженную карту с "своей"
                    frukto_result = compare_cards_fruito10_min_usage(play_card_key, my_card_key)
                    if frukto_result is None:
                        print("Нет комбинации, суммой дающей 10.")
                    else:
                        cond, group_value, card1_part , card2_part = frukto_result
                        print(frukto_result)
                        #print(len(card1_part[0][0]))
                        if cond == "color":
                            print(f"--- Минимальная комбинация по цвету '{group_value}' ---")
                            #(BLACK, RED, YELLOW, GREEN, CYAN, BLUE, PURPLE, WHITE) 
                            if group_value=='Фиолетовый':
                                pico.apply(0,'PURPLE')
                            elif group_value=='Голубой':
                                pico.apply(0,'CYAN')
                            elif group_value=='Зеленый':
                                pico.apply(0,'GREEN')
                            elif group_value=='Желтый':
                                pico.apply(0,'YELLOW')
                            elif group_value=='Красный':
                                pico.apply(0,'RED')
                            elif group_value=='Оранжевый':
                                pico.apply(0,'WHITE')
                        else:
                            print(f"--- Минимальная комбинация по типу '{group_value}' ---")

                        print("Из 1-й карты:", card1_part)
                        print("Из 2-й карты:", card2_part)
                        total_sum = sum(x[1] for x in card1_part) + sum(x[1] for x in card2_part)
                        print("Суммарное количество фруктов =", len(card1_part) + len(card2_part))
                        print("Сумма =", total_sum)
                    
                        if was_clicked:
                            self.match_found = True
                            pos_z = pos_z - self.z_step
                            self.pos_card_center = (pos_x, pos_y, pos_z)
                            pico.apply(0,'GREEN')
                            
                            self.score_robot += 1
                            self.scoreUpdated.emit(self.score_robot, self.score_human)

                            img_play_card_show = draw_matching_boxes(img_card_detected, play_card_results, card1_part)
                            img_play_card_objs = extract_and_concatenate(img_card_detected, play_card_results, card1_part)
                            img_my_card_show   = draw_matching_boxes(img_my_card, my_card_results, card2_part)
                            img_my_card_objs   = extract_and_concatenate(img_my_card, my_card_results, card2_part)

                            self.newFrameMyCard.emit(img_my_card_show)
                            self.newFramePlayCard.emit(img_play_card_show)
                            self.newFrameLastComboRobot.emit(img_my_card_objs)
                            self.newFrameLastComboField.emit(img_play_card_objs)

                            #подождать пока говорит в другом потоке
                            while isSpeaking:
                                time.sleep(0.1)
                            
                            # сказать что нашел
                            Say_phraze("found")
                            if cond=="color":
                                Say_object_color(group_value)
                                for obj in card1_part:
                                    label_color, digit= obj
                                    # Преобразование строки в кортеж
                                    tuple_data = literal_eval(label_color)  # Преобразуем в кортеж (0, 'Оранжевый')
                                    # Распаковка кортежа
                                    class_label, color = tuple_data
                                    Say_object_class(class_label)    

                                for obj in card2_part:
                                    label_color, digit= obj
                                    # Преобразование строки в кортеж
                                    tuple_data = literal_eval(label_color)  # Преобразуем в кортеж (0, 'Оранжевый')
                                    # Распаковка кортежа
                                    class_label, color = tuple_data
                                    Say_object_class(class_label)   
                            elif cond=="type":
                                Say_object_class(group_value)
                                for obj in card1_part:
                                    label_color, digit= obj
                                    # Преобразование строки в кортеж
                                    tuple_data = literal_eval(label_color)  # Преобразуем в кортеж (0, 'Оранжевый')
                                    # Распаковка кортежа
                                    class_label, color = tuple_data
                                    Say_object_color(color)    
                                for obj in card2_part:
                                    label_color, digit= obj
                                    # Преобразование строки в кортеж
                                    tuple_data = literal_eval(label_color)  # Преобразуем в кортеж (0, 'Оранжевый')
                                    # Распаковка кортежа
                                    class_label, color = tuple_data
                                    Say_object_color(color)   
                            #здесь озвучка закончится

                            # Перенос карты "себе"
                            self.move_card(self.pos_card_center,"robot")
                            self.pos_robots_card_down=tuple(np.add(self.xyz_robots_card_down, (0,0,self.step_z)))
                            set_position(self.pos_camera_home)
                            time.sleep(2)

                            # Запоминаем свою карту
                            img_my_card = img_card_detected
                            my_card_results = play_card_results
                            my_card_key = play_card_key

                            self.newFrameMyCard.emit(img_my_card)
                            img_play_card = np.ones_like(img_my_card)
                            self.newFramePlayCard.emit(img_play_card)
                            time_last_predict = time.time()
                            t_pause = 2


class WaitWorker(QObject):
    newFrameCamera = pyqtSignal(object)
    global is_online

    def __init__(self,is_online):
        super().__init__()
        self._running = True
        self.is_online=is_online

    def stop(self):
        self._running = False

    def run(self):
        print("[WaitWorker] Цикл WAIT (пауза).")
        global was_dbl_clicked
        global isSpeaking

        if self.is_online:
            try:
                isSpeaking=True
                print("Dialog start")
                #phraza="Слушаю Вас"
                Say_phraze("listen")
                #generate_speech(phraza)
                text=recognize_speech()
                text=text.lower()
                print("Пользователь сказал: "+text)
                if ("счет" in text) or ("счёт" in text):
                    if self.score_robot>self.score_human:
                        phraza="Сейчас счёт "+str(self.score_robot)+" : "+str(self.score_human)+" в мою пользу"
                    elif self.score_robot==self.score_human:
                        phraza="Сейчас ничья "+str(self.score_robot)+" : "+str(self.score_human)
                    else:
                        phraza="Сейчас счёт "+str(self.score_human)+" : "+str(self.score_robot)+" в твою пользу"
                    generate_speech(phraza)
                if ("время" in text) or (" час" in text):
                    # Local time has date and time
                    t = time.localtime()
                    # Extract the time part
                    #current_time = time.strftime("%H:%M:%S", t)
                    hour_minutes=time.strftime("%H:%M",t)
                    #minutes=time.strftime("%M",t)
                    phraza="текущее время "+hour_minutes
                    generate_speech(phraza)
                if ("кто" in text) and (("автор" in text) or ("сделал" in text)):
                    phraza="Автор этого проекта Егор Каржавин, ученик девятого А класса Лицея Иннополис"
                    generate_speech(phraza)
            except:
                print("Yandex speech error")
            isSpeaking=False
       
        was_dbl_clicked=False
        
        while self._running:
            frame = stream.read()
            self.newFrameCamera.emit(frame)

        print("[WaitWorker] Завершен")


# ------------ Основной класс с машиной состояний ------------
class FruktoStateMachine(QObject):
    def __init__(self, is_online):
        super().__init__()
        # 1) Создаём приложение и GUI
        self.app = QApplication(sys.argv)
        self.is_online = is_online
        self.gui = MainApp(self.app, self.is_online)

        # Сигналы от кнопок GUI
        self.gui.start_signal.connect(self.on_start_button)
        self.gui.pause_signal.connect(self.on_pause_button)
        self.gui.stop_signal.connect(self.on_stop_button)
        self.gui.close_signal.connect(self.on_close_window)

        self.gui.apply_settings()
        
        self.score_robot = 0
        self.score_human = 0

        # 2) Описываем состояния и переходы
        states = [
            State(name='begin', on_enter=['on_enter_begin'], on_exit=['on_exit_begin']),
            State(name='play',  on_enter=['on_enter_play'],  on_exit=['on_exit_play']),
            State(name='wait',  on_enter=['on_enter_wait'],  on_exit=['on_exit_wait']),
            State(name='stop',  on_enter=['on_enter_stop'],  on_exit=['on_exit_stop']),
            State(name='END',  on_enter=['on_enter_END'])
        ]

        transitions = [
            {'trigger': 'begin_to_play', 'source': 'begin', 'dest': 'play'},
            {'trigger': 'begin_to_END',  'source': 'begin', 'dest': 'END'},

            {'trigger': 'play_to_wait', 'source': 'play', 'dest': 'wait'},
            {'trigger': 'play_to_stop', 'source': 'play', 'dest': 'stop'},
            {'trigger': 'play_to_END',  'source': 'play', 'dest': 'END'},

            {'trigger': 'wait_to_play', 'source': 'wait', 'dest': 'play'},
            {'trigger': 'wait_to_stop', 'source': 'wait', 'dest': 'stop'},
            {'trigger': 'wait_to_END',  'source': 'wait', 'dest': 'END'},

            {'trigger': 'stop_to_play', 'source': 'stop', 'dest': 'play'},
            {'trigger': 'stop_to_END',  'source': 'stop', 'dest': 'END'},
        ]

        self.machine = Machine(
            model=self,
            states=states,
            transitions=transitions,
            initial='begin',
            ignore_invalid_triggers=True
        )

        QCoreApplication.instance().aboutToQuit.connect(self.cleanup_threads)
        self.on_enter_begin()

        sys.exit(self.app.exec())

    # ---------- Триггеры по кнопкам GUI ----------
    def on_start_button(self):
        cur_state = self.state
        if cur_state == 'begin':
            self.begin_to_play()
        elif cur_state == 'wait':
            self.wait_to_play()
        elif cur_state == 'stop':
            self.stop_to_play()

    def on_pause_button(self, is_paused):
        cur_state = self.state
        if cur_state == 'play':
            self.play_to_wait()
        elif cur_state == 'wait':
            self.wait_to_play()

    def on_stop_button(self):
        cur_state = self.state
        if cur_state == 'play':
            self.play_to_stop()
        elif cur_state == 'wait':
            self.wait_to_stop()

    def on_close_window(self):
        cur_state = self.state
        if cur_state == 'begin':
            self.begin_to_END()
        elif cur_state == 'play':
            self.play_to_END()
        elif cur_state == 'wait':
            self.wait_to_END()
        elif cur_state == 'stop':
            self.stop_to_END()
        # Закрытие приложения чуть позже
        QTimer.singleShot(1000, self.app.quit)

    # ---------- Методы изменения состояний State Machine ----------
    def on_enter_begin(self):
        print("Состояние BEGIN")
        self._thread_begin = QThread()
        self._worker_begin = BeginWorker()

        self._worker_begin.newFrameCamera.connect(self.show_camera_frame_begin)
        self._worker_begin.newFrameZone.connect(self.show_zone_frame_begin)

        self._worker_begin.requestSlider1Value.connect(self.provide_slider1_value)
        self._worker_begin.requestSlider2Value.connect(self.provide_slider2_value)

        self._worker_begin.moveToThread(self._thread_begin)
        self._thread_begin.started.connect(self._worker_begin.run)
        self._thread_begin.start()

    def provide_slider1_value(self):
        h1_min = self.gui.window.findChild(QtWidgets.QSlider, "hSliderH1min").value()
        h1_max = self.gui.window.findChild(QtWidgets.QSlider, "hSliderH1max").value()
        s1_min = self.gui.window.findChild(QtWidgets.QSlider, "hSliderS1min").value()
        s1_max = self.gui.window.findChild(QtWidgets.QSlider, "hSliderS1max").value()
        v1_min = self.gui.window.findChild(QtWidgets.QSlider, "hSliderV1min").value()
        v1_max = self.gui.window.findChild(QtWidgets.QSlider, "hSliderV1max").value()
        slider = (h1_min, h1_max, s1_min, s1_max, v1_min, v1_max)
        self._worker_begin.on_slider1_value_received(slider)

    def provide_slider2_value(self):
        h2_min = self.gui.window.findChild(QtWidgets.QSlider, "hSliderH2min").value()
        h2_max = self.gui.window.findChild(QtWidgets.QSlider, "hSliderH2max").value()
        s2_min = self.gui.window.findChild(QtWidgets.QSlider, "hSliderS2min").value()
        s2_max = self.gui.window.findChild(QtWidgets.QSlider, "hSliderS2max").value()
        v2_min = self.gui.window.findChild(QtWidgets.QSlider, "hSliderV2min").value()
        v2_max = self.gui.window.findChild(QtWidgets.QSlider, "hSliderV2max").value()
        slider = (h2_min, h2_max, s2_min, s2_max, v2_min, v2_max)
        self._worker_begin.on_slider2_value_received(slider)

    def on_exit_begin(self):
        print("Выходим из begin")
        if hasattr(self, '_thread_begin'):
            self._worker_begin.stop()
            self._thread_begin.quit()
            self._thread_begin.wait()

    def on_enter_play(self):
        print("Состояние PLAY")
        # Считываем конфигурацию из SettingsManager
        config = self.gui.settings_manager.read_table(self.gui.settings_table)

        # Выполняем init-блок (запуск распознавания, получение карты)
        self.do_init_block(config)
        time.sleep(1)

        # Запускаем рабочий поток для "play"
        self._thread_play = QThread()
        self._worker_play = PlayWorker(config=config)

        self._worker_play.newFrameCamera.connect(self.show_camera_frame_play)
        self._worker_play.newFrameMyCard.connect(self.show_MyCard_frame_play)
        self._worker_play.newFramePlayCard.connect(self.show_PlayCard_frame_play)
        self._worker_play.newFrameLastComboRobot.connect(self.show_LastComboRobot_frame_play)
        self._worker_play.newFrameLastComboField.connect(self.show_LastComboField_frame_play)

        self._worker_play.scoreUpdated.connect(self.update_scores)

        self._worker_play.moveToThread(self._thread_play)
        self._thread_play.started.connect(self._worker_play.run)
        self._thread_play.start()

    def update_scores(self, score_robot, score_human):
        self.score_robot = score_robot
        self.score_human = score_human
        print(f"Обновлённый счёт - Робот: {self.score_robot}, Человек: {self.score_human}")
        self.gui.update_lcd_value(score_human, "lcdNumberPlayer") 
        self.gui.update_lcd_value(score_robot, "lcdNumberRobot") 
        # Если нужен автоматический выход по достижению определённого счёта:
        if (self.score_robot + self.score_human) > 2:
            print("Игра окончена, переход в состояние STOP")
            self.gui.stop_pressed()

    def on_exit_play(self):
        print("Выходим из play")
        if hasattr(self, '_thread_play'):
            self._worker_play.stop()
            self._thread_play.quit()
            self._thread_play.wait()

    def on_enter_wait(self):
        print("Состояние WAIT")
        self._thread_wait = QThread()
        self._worker_wait = WaitWorker(self.is_online)
        self._worker_wait.newFrameCamera.connect(self.show_camera_frame_wait)

        self._worker_wait.moveToThread(self._thread_wait)
        self._thread_wait.started.connect(self._worker_wait.run)
        self._thread_wait.start()

    def on_exit_wait(self):
        print("Выходим из wait")
        if hasattr(self, '_thread_wait'):
            self._worker_wait.stop()
            self._thread_wait.quit()
            self._thread_wait.wait()

    def on_enter_stop(self):
        print("Состояние STOP")
        print("Robot", self.score_robot)
        print("Human", self.score_human)
        #говорим прощальные фразы
        if self.is_online: 
            if self.score_robot>self.score_human:
                phraza="игра закончилась со счётом "+str(self.score_robot)+" : "+str(self.score_human)+" в мою пользу"
            elif self.score_robot==self.score_human:
                phraza="игра закончилась в ничью "+str(self.score_robot)+" : "+str(self.score_human)
            else:
                phraza="игра закончилась со счётом "+str(self.score_human)+" : "+str(self.score_robot)+" в твою пользу"
            generate_speech(phraza)

        if self.score_robot>self.score_human:
            Say_phraze("robot_win")
            print("robot_win")
        elif self.score_robot==self.score_human:
            Say_phraze("nobody")
            print("nobody")
        else:
            Say_phraze("human_win")
            print("human_win")

        print("[PlayWorker] Завершен")

    def on_exit_stop(self):
        print("Выходим из stop")

    def on_enter_END(self):
        print("Состояние END. Завершение работы.")
        self.cleanup_threads()

    def cleanup_threads(self):
        print("Начинается завершение потоков в cleanup_threads()")
        if hasattr(self, '_thread_begin') and self._thread_begin.isRunning():
            self._worker_begin.stop()
            self._thread_begin.quit()
            if not self._thread_begin.wait(5000):
                print("Thread _thread_begin не завершился вовремя, принудительно завершаем.")
                self._thread_begin.terminate()
                self._thread_begin.wait()

        if hasattr(self, '_thread_play') and self._thread_play.isRunning():
            self._worker_play.stop()
            self._thread_play.quit()
            if not self._thread_play.wait(5000):
                print("Thread _thread_play не завершился вовремя, принудительно завершаем.")
                self._thread_play.terminate()
                self._thread_play.wait()

        if hasattr(self, '_thread_wait') and self._thread_wait.isRunning():
            self._worker_wait.stop()
            self._thread_wait.quit()
            if not self._thread_wait.wait(5000):
                print("Thread _thread_wait не завершился вовремя, принудительно завершаем.")
                self._thread_wait.terminate()
                self._thread_wait.wait()

    # ---------- Слоты приёма кадров (из воркеров) ----------
    def show_camera_frame_begin(self, frame):
        self.gui.load_image(frame, "graphicsViewCamera")

    def show_zone_frame_begin(self, zone_img):
        self.gui.load_image(zone_img, "zone_debugView")

    def show_camera_frame_play(self, frame):
        self.gui.load_image(frame, "graphicsViewCamera")

    def show_MyCard_frame_play(self, frame):
        self.gui.load_image(frame, "graphicsMyCard")

    def show_PlayCard_frame_play(self, frame):
        self.gui.load_image(frame, "graphicsPlayCard")

    def show_LastComboRobot_frame_play(self, frame):
        self.gui.load_image(frame, "graphicsLastComboRobot")

    def show_LastComboField_frame_play(self, frame):
        self.gui.load_image(frame, "graphicsLastComboField")

    def show_camera_frame_wait(self, frame):
        self.gui.load_image(frame, "graphicsViewCamera")

    # ---------- init-блок (который вызывается при переходе begin -> play / stop -> play) ----------
    def do_init_block(self, config):
        global img_my_card, my_card_results, my_card_key, img_play_card

        self.config = config
        self.pos_camera_home_start= config.get("pos_camera_home_start", [-150, 0, 180])
        set_position(self.pos_camera_home_start)
        
        print("=== do_init_block: выполняем INIT-блок ===")
        usl = False
        time_last_predict = time.time()
        n_detected = 0

        while not usl:
            time.sleep(0.1)
            frame = stream.read()
            # frame = cv2.imread("Pictures/Screenshot from 2025-02-16 15-33-40.png")
            self.gui.load_image(frame, "graphicsViewCamera")

            if (time.time() - time_last_predict) > 1:
                img_card_detected, play_card_results, play_card_key, card_box, zone_box = predict_card(frame)
                time_last_predict = time.time()
                self.gui.load_image(img_card_detected, "graphicsMyCard")
            else:
                img_card_detected = None

            if img_card_detected is not None:
                n_detected = len(play_card_results)

            # Выходим из цикла, если нашли 8 объектов
            if n_detected == 8:
                usl = True

        img_my_card = img_card_detected
        my_card_results = play_card_results
        my_card_key = play_card_key

        print(my_card_results)
        print(my_card_key)

        self.gui.load_image(img_my_card, "graphicsMyCard")
        img_play_card = np.ones_like(img_my_card)
        self.gui.load_image(img_play_card, "graphicsPlayCard")
        print("=== init-блок завершён ===")


if __name__ == "__main__":
    was_clicked = False
    was_dbl_clicked = False
    isSpeaking=False
    is_online=checkInternetHttplib()
    # здесь стратануть камеру
    stream = camera()
    stream.start()
    machine = FruktoStateMachine(is_online)
