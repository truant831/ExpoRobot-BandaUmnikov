# GUI_fsm.py
import sys
import time
import cv2
import numpy as np

from transitions import Machine, State
from PySide6 import QtWidgets
from PySide6.QtWidgets import QApplication 
from PySide6.QtCore import QThread, QObject, Signal, QTimer, QCoreApplication

from GUI_class import MainApp
from libs.video_predict_frukto2 import predict_card, compare_cards_fruito10_min_usage,  draw_matching_boxes, get_card_key, extract_and_concatenate, find_play_zone_dynamic
# from libs.play_sounds import Say_phraze, ...
# и т.д.

# ------------ Пример рабочих потоков ------------
class BeginWorker(QObject):
    newFrameCamera = Signal(object)
    newFrameZone = Signal(object)

    requestSlider1Value = Signal()
    requestSlider2Value = Signal()
    #newMessage1 = Signal(str)
    #newMessage2 = Signal(str)

    def __init__(self):
        super().__init__()
        self._running = True
        # Добавляем значения по умолчанию, чтобы не было ошибки,
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
                frame = cv2.imread("Pictures/Screenshot from 2025-02-16 15-33-40.png")
                time.sleep(0.02)
                # Запросим значения ВСЕХ слайдеров у GUI
                self.requestSlider1Value.emit()
                self.requestSlider2Value.emit()

                #отдадим картинку с камеры в GUI
                self.newFrameCamera.emit(frame)
                
                #если захотим в основной поток отдать сообщение о том, что приняли
                #self.newMessage1.emit(f"Текущее значение слайдера = {self.slider1_value}")
                #self.newMessage2.emit(f"Текущее значение слайдера = {self.slider2_value}")
                
                HSV_limit1=self.slider1_value
                HSV_limit2=self.slider2_value

                #print(HSV_limit1)
                #print(HSV_limit2)

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
    newFrameCamera = Signal(object)

    newFrameMyCard = Signal(object)
    newFramePlayCard= Signal(object)
    newFrameLastComboRobot= Signal(object)
    newFrameLastComboField= Signal(object)

    scoreUpdated = Signal(int, int)  # (score_robot, score_human)

    def __init__(self, config):
        super().__init__()
        self._running = True
        self.isSpeaking=False
        self.match_found=False
        # Получаем всю конфигурацию
        self.config = config

        # Извлекаем необходимые переменные из конфигурации, второй параметр нужен как дефолт
        self.pos_card_center     = config.get("pos_card_center", [0, 210, 0])
        self.pos_camera_home     = config.get("pos_camera_home", [0, 160, 180])
        self.xyz_robots_card_down= config.get("xyz_robots_card_down", [-210, 0, 10])
        self.xyz_human_card_down = config.get("xyz_human_card_down", [210, 0, 30])
        self.pos_camera_home_start= config.get("pos_camera_home_start", [-150, 0, 180])
        self.pos_camera_home_fuman= config.get("pos_camera_home_fuman", [150, 0, 180])
        self.z_step              = config.get("z_step", 0.45)

        print("[PlayWorker] Загруженные настройки:")
        print("pos_card_center:", self.pos_card_center)
        print("pos_camera_home:", self.pos_camera_home)
        print("xyz_robots_card_down:", self.xyz_robots_card_down)
        print("xyz_human_card_down:", self.xyz_human_card_down)
        print("pos_camera_home_start:", self.pos_camera_home_start)
        print("pos_camera_home_fuman:", self.pos_camera_home_fuman)
        print("z_step:", self.z_step)

        # Инициализируем локальные переменные для счёта
        self.score_robot=0
        self.score_human=0

    def stop(self):
        self._running = False

    def run(self):
        print("[PlayWorker] Основной цикл PLAY (после init-блока).")
        global img_my_card, my_card_results, my_card_key, img_play_card #глобальные, т.к. разделили на потоки программу
        global was_clicked
        global is_online

        time_last_predict=time.time()
        n_detected=0
        pos_x=0
        pos_y=0
        pos_z=0
        t_pause=0

        while self._running:
            time.sleep(0.1)
            frame = cv2.imread("Pictures/Screenshot from 2025-02-16 15-34-07.png")
            self.newFrameCamera.emit(frame)
            time.sleep(0.1)  # Имитируем задержку (заменить на захват камеры)
    
            if was_clicked and card_x>0 and card_y>0: #обновляется через отдельный поток и callback 
                # перенос карты фумену
                #добавим балл человеку
                self.score_human+=1 
                # Эмитируем сигнал с новыми значениями счёта:
                self.scoreUpdated.emit(self.score_robot, self.score_human) 

                pos_x, pos_y, pos_z= self.pos_card_center
                pos_x=int(0+(card_x-340)/4) #recalc x where we will take card based on card center on camera
                pos_y=int(210-(card_y-250)/4) #recalc y where we will take card based on card center on camera
                pos_z=pos_z-self.z_step
                self.pos_card_center=(pos_x, pos_y, pos_z)
                was_clicked=False
                ####move_card(self.pos_card_center,"human")
                self.pos_card_center=tuple(np.subtract(self.pos_card_center, (0,0,0.45)))
                ####set_position(self.pos_camera_home)
                time.sleep(1)
            else:
                self.match_found=False
                #обнуляем щелчок кнопки, чтобы потом смотреть был ли он за время одной итерации цикла
                was_clicked=False
                # Чтение кадра
                ### frame = stream.read()
                frame=cv2.imread("Pictures/Screenshot from 2025-02-16 15-34-07.png") ### убрать при возврате на камеру
                time.sleep(1.5) #временная пауза, убрать при возврате на камеру

                #запускаем распознавание не чаще чем раз в 0.5+t_pause сек
                if (time.time()-time_last_predict)>(0.5+t_pause):
                    img_card_detected, play_card_results, play_card_key, card_box, zone_box = predict_card(frame)
                    time_last_predict=time.time()
                    t_pause=0
                else:
                    img_card_detected=None

                if img_card_detected is not None:
                    n_detected = len(play_card_results)
                    #координаты центра карты в координатах камеры, относительно Кадра с прямоугольником игровой зоны
                    (x,y,w,h)=card_box
                    card_x=int(x+w/2)
                    card_y=int(y+h/2)
                    print("Card center " +str(card_x)+" , "+str(card_y))
                    #считам координаты центра карты в кординатах реального мира для руки
                    pos_x, pos_y, pos_z= self.pos_card_center
                    pos_x=int(0+(card_x-340)/4) #recalc x where we will take card based on card center on camera
                    pos_y=int(210-(card_y-250)/4) #recalc y where we will take card based on card center on camera

                    self.newFramePlayCard.emit(img_card_detected)
                    time.sleep(4.5) #временная пауза

                    #проверить совпадения в картинках своей карты и карты поля
                    frukto_result = compare_cards_fruito10_min_usage(play_card_key, my_card_key) # сравнить свою карту и карту поля
                    if frukto_result is None:
                        print("Нет комбинации, суммой дающей 10.")
                    else:
                        cond, group_value, card1_part , card2_part = frukto_result
                        print(frukto_result)
                        #print(len(card1_part[0][0]))
                        if cond == "color":
                            print(f"--- Минимальная комбинация по цвету '{group_value}' ---")
                            #(BLACK, RED, YELLOW, GREEN, CYAN, BLUE, PURPLE, WHITE) 
                            # if group_value=='Фиолетовый':
                            #     pico.apply(0,'PURPLE')
                            # elif group_value=='Голубой':
                            #     pico.apply(0,'CYAN')
                            # elif group_value=='Зеленый':
                            #     pico.apply(0,'GREEN')
                            # elif group_value=='Желтый':
                            #     pico.apply(0,'YELLOW')
                            # elif group_value=='Красный':
                            #     pico.apply(0,'RED')
                            # elif group_value=='Оранжевый':
                            #     pico.apply(0,'WHITE')
                        else:
                            print(f"--- Минимальная комбинация по типу '{group_value}' ---")

                        print("Из 1-й карты:", card1_part)
                        print("Из 2-й карты:", card2_part)
                        total_sum = sum(x[1] for x in card1_part) + sum(x[1] for x in card2_part)
                        print("Суммарное количество фруктов =", len(card1_part) + len(card2_part))
                        print("Сумма =", total_sum)
                    
                    print(n_detected)
                    print(was_clicked)
                    # перенос карты себе если нашли совпадение и человек не успел
                    if len(frukto_result) > 0 and n_detected==8 and was_clicked==False: 
                        self.match_found=True
                        pos_z=pos_z-self.z_step #вычтем Z чтобы в след раз брать карту чуть ниже
                        self.pos_card_center=(pos_x, pos_y, pos_z)
                        ### pico.apply(0,'GREEN')
                        
                        #добавим себе балл
                        self.score_robot+=1    
                        # Эмитируем сигнал с новыми значениями счёта:
                        self.scoreUpdated.emit(self.score_robot, self.score_human) 
                                
                        #создаем копии картинок с обведенной рамкой обнаруженного объекта
                        img_play_card_show=draw_matching_boxes(img_card_detected, play_card_results, card1_part)
                        img_play_card_objs=extract_and_concatenate(img_card_detected,play_card_results,card1_part)
                        img_my_card_show=draw_matching_boxes(img_my_card, my_card_results, card2_part)
                        img_my_card_objs=extract_and_concatenate(img_my_card, my_card_results,card2_part)
                        #можно вот так в одну объединить картинку и потом ее показать, если не под одной, но надо еще их одиноковой высоты сделать перед объединением
                        #Hori = np.concatenate((img_my_card_show, img_play_card_show), axis=1) 
                        #cv2.imshow('Match!', Hori)    

                        #показываем по одной
                        self.newFrameMyCard.emit(img_my_card_show)
                        self.newFramePlayCard.emit(img_play_card_show)
                        self.newFrameLastComboRobot.emit(img_my_card_objs)
                        self.newFrameLastComboField.emit(img_play_card_objs)

                        #здесь будет озвучка

                        #здесь озвучка закончится

                        # перенос карты себе
                        #move_card(pos_card_center,"robot")
                        #pos_robots_card_down=tuple(np.add(xyz_robots_card_down, (0,0,0.45)))
                        #set_position(pos_camera_home)
                        time.sleep(2) #пауза чтобы робот доехал

                        # запоминаем свою карту
                        img_my_card=img_card_detected
                        my_card_results=play_card_results
                        my_card_key=play_card_key

                        self.newFrameMyCard.emit(img_my_card)
                        img_play_card=np.ones_like(img_my_card)
                        self.newFramePlayCard.emit(img_play_card)
                        time_last_predict=time.time()
                        t_pause=2   
        

class WaitWorker(QObject):
    newFrameCamera = Signal(object)

    def __init__(self):
        super().__init__()
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        print("[WaitWorker] Цикл WAIT (пауза).")
        frame = cv2.imread("Pictures/Screenshot from 2025-02-16 15-33-40.png")

        while self._running:
            time.sleep(0.05)
            self.newFrameCamera.emit(frame)

        print("[WaitWorker] Завершен")


# ------------ Основной класс с машиной состояний ------------
class FruktoStateMachine(QObject):
    def __init__(self):
        super().__init__()

        # 1) Создаем приложение и GUI
        self.app = QApplication(sys.argv)
        self.is_online=False #доделать на функцию из либы
        self.gui = MainApp(self.app, self.is_online)

        # Сигналы от кнопок GUI
        self.gui.start_signal.connect(self.on_start_button)
        self.gui.pause_signal.connect(self.on_pause_button)
        self.gui.stop_signal.connect(self.on_stop_button)
        self.gui.close_signal.connect(self.on_close_window)

        self.gui.apply_settings()
        
        # Инициализируем локальные переменные для счёта
        self.score_robot=0
        self.score_human=0

        # 2) Описываем состояния и переходы
        states = [
            State(name='begin', on_enter=['on_enter_begin'], on_exit=['on_exit_begin']),
            State(name='play',  on_enter=['on_enter_play'],  on_exit=['on_exit_play']),
            State(name='wait',  on_enter=['on_enter_wait'],  on_exit=['on_exit_wait']),
            State(name='stop',  on_enter=['on_enter_stop'],  on_exit=['on_exit_stop']),
            State(name='END',   on_enter=['on_enter_END'])
        ]

        transitions = [
            # begin -> play / STOP
            # "after": 'do_init_block' -- вызываем init-блок сразу после перехода
            {'trigger': 'begin_to_play', 'source': 'begin', 'dest': 'play'},
            {'trigger': 'begin_to_END',  'source': 'begin', 'dest': 'END'},

            # play -> wait / stop / STOP
            {'trigger': 'play_to_wait', 'source': 'play', 'dest': 'wait'},
            {'trigger': 'play_to_stop', 'source': 'play', 'dest': 'stop'},
            {'trigger': 'play_to_END',  'source': 'play', 'dest': 'END'},

            # wait -> play / stop / STOP
            # при wait -> play уже НЕ нужен init-блок
            {'trigger': 'wait_to_play', 'source': 'wait', 'dest': 'play'},
            {'trigger': 'wait_to_stop', 'source': 'wait', 'dest': 'stop'},
            {'trigger': 'wait_to_END',  'source': 'wait', 'dest': 'END'},

            # stop -> play / STOP
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
        # Вместо немедленного завершения приложения,
        # отложим вызов app.quit() на 100 мс:
        QTimer.singleShot(1000, self.app.quit)

    # ---------- Методы изменения состояний State Machine ----------
    def on_enter_begin(self):
        print("Состояние BEGIN")
        self._thread_begin = QThread()
        self._worker_begin = BeginWorker()
        # Сигналы -> слоты
        self._worker_begin.newFrameCamera.connect(self.show_camera_frame_begin)
        self._worker_begin.newFrameZone.connect(self.show_zone_frame_begin)

        self._worker_begin.requestSlider1Value.connect(self.provide_slider1_value)
        self._worker_begin.requestSlider2Value.connect(self.provide_slider2_value)
        #self._worker_begin.newMessage1.connect(lambda text: print("[MSG1]", text))
        #self._worker_begin.newMessage2.connect(lambda text: print("[MSG2]", text))

        #создаем потоки воркера 
        self._worker_begin.moveToThread(self._thread_begin)
        self._thread_begin.started.connect(self._worker_begin.run)
        self._thread_begin.start()

    def provide_slider1_value(self):
        # Этот слот вызывается в главном потоке, значит
        # безопасно обратиться к GUI 
        h1_min = self.gui.window.findChild(QtWidgets.QSlider, "hSliderH1min").value()
        h1_max = self.gui.window.findChild(QtWidgets.QSlider, "hSliderH1max").value()
        s1_min = self.gui.window.findChild(QtWidgets.QSlider, "hSliderS1min").value()
        s1_max = self.gui.window.findChild(QtWidgets.QSlider, "hSliderS1max").value()
        v1_min = self.gui.window.findChild(QtWidgets.QSlider, "hSliderV1min").value()
        v1_max = self.gui.window.findChild(QtWidgets.QSlider, "hSliderV1max").value()

        slider = h1_min, h1_max, s1_min, s1_max, v1_min, v1_max
        self._worker_begin.on_slider1_value_received(slider)

    def provide_slider2_value(self):
        # Этот слот вызывается в главном потоке, значит
        # безопасно обратиться к GUI
        h2_min = self.gui.window.findChild(QtWidgets.QSlider, "hSliderH2min").value()
        h2_max = self.gui.window.findChild(QtWidgets.QSlider, "hSliderH2max").value()
        s2_min = self.gui.window.findChild(QtWidgets.QSlider, "hSliderS2min").value()
        s2_max = self.gui.window.findChild(QtWidgets.QSlider, "hSliderS2max").value()
        v2_min = self.gui.window.findChild(QtWidgets.QSlider, "hSliderV2min").value()
        v2_max = self.gui.window.findChild(QtWidgets.QSlider, "hSliderV2max").value()

        slider = h2_min, h2_max, s2_min, s2_max, v2_min, v2_max
        self._worker_begin.on_slider2_value_received(slider)

    def on_exit_begin(self):
        print("Выходим из begin")
        if hasattr(self, '_thread_begin'):
            self._worker_begin.stop()
            self._thread_begin.quit()
            self._thread_begin.wait()

    def on_enter_play(self):
        print("Состояние PLAY")
        # Выполняем init-блок синхронно; это заблокирует главный поток, так что, если работа занимает много времени, UI может «подвиснуть».
        self.do_init_block()
        time.sleep(1)
        # После завершения init-блока запускаем основной рабочий поток

        # Считываем всю конфигурацию из SettingsManager из таблицы GUI, не из файла!
        config = self.gui.settings_manager.read_table(self.gui.settings_table)

        # Запускаем основной рабочий поток для Play
        self._thread_play = QThread()
        self._worker_play = PlayWorker(config=config)

        #Подписки на все события обновления картинок окошек. лайк, репост))
        self._worker_play.newFrameCamera.connect(self.show_camera_frame_play)

        self._worker_play.newFrameMyCard.connect(self.show_MyCard_frame_play)
        self._worker_play.newFramePlayCard.connect(self.show_PlayCard_frame_play)
        self._worker_play.newFrameLastComboRobot.connect(self.show_LastComboRobot_frame_play)
        self._worker_play.newFrameLastComboField.connect(self.show_LastComboField_frame_play)

        # Подписываемся на обновления счёта:
        self._worker_play.scoreUpdated.connect(self.update_scores)

        self._worker_play.moveToThread(self._thread_play)
        self._thread_play.started.connect(self._worker_play.run)
        self._thread_play.start()

    def update_scores(self, score_robot, score_human):
        # Этот слот вызывается в главном потоке.
        self.score_robot = score_robot
        self.score_human = score_human
        print(f"Обновлённый счёт - Робот: {self.score_robot}, Человек: {self.score_human}")
        self.gui.update_lcd_value(score_human,"lcdNumberPlayer") 
        self.gui.update_lcd_value(score_robot,"lcdNumberRobot") 
        #условие что закончились карты
        if (self.score_robot+self.score_human)>2: #сколько карт разыграть
            print("Игра окончена, переход в состояние STOP")
            #self.play_to_stop()
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
        self._worker_wait = WaitWorker()
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
        #TODO сделать передачу параметра is_online
        # if self.is_online: 
        #     if score_robot>score_human:
        #         phraza="игра закончилась со счётом "+str(score_robot)+" : "+str(score_human)+" в мою пользу"
        #     elif score_robot==score_human:
        #         phraza="игра закончилась в ничью "+str(score_robot)+" : "+str(score_human)
        #     else:
        #         phraza="игра закончилась со счётом "+str(score_human)+" : "+str(score_robot)+" в твою пользу"
        #     generate_speech(phraza)

        # if self.score_robot>self.score_human:
        #     Say_phraze("robot_win")
        #     print("robot_win")
        # elif self.score_robot==self.score_human:
        #     Say_phraze("nobody")
        #     print("nobody")
        # else:
        #     Say_phraze("human_win")
        #     print("human_win")
        print("[PlayWorker] Завершен")

    def on_exit_stop(self):
        print("Выходим из stop")

    def on_enter_END(self):
        print("Состояние STOP. Завершение работы.")
        #очистим потоки, потом в on_close_window закроем gui
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

    # ---------- Слоты для приёма кадров (из воркеров) ----------
    def show_camera_frame_begin(self, frame):
        self.gui.load_image(frame, "graphicsViewCamera")

    def show_zone_frame_begin(self, zone_img):
        self.gui.load_image(zone_img, "zone_debugView")

    def show_camera_frame_play(self, frame):
        self.gui.load_image(frame, "graphicsViewCamera")
    
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

    # ---------- init-блок ----------
    def do_init_block(self):
        """
        Этот метод вызывается автоматически после перехода:
         - begin -> play
         - stop -> play
        (согласно настройке 'after': 'do_init_block' в transitions)
        Здесь вставляем ваш код инициализации (строки 109–166).
        """
        global img_my_card, my_card_results, my_card_key, img_play_card #глобальные, т.к. разделили на потоки программу

        print("=== do_init_block: выполняем INIT-блок (строки 109..166 старой проги) ===")
        usl=False
        time_last_predict=time.time()
        n_detected=0
        while not usl:
            #frame = stream.read()  # Убрать при возврате на камеру
            time.sleep(0.1)  # Имитируем задержку (заменить на захват камеры)
            frame = cv2.imread("Pictures/Screenshot from 2025-02-16 15-33-40.png")
            self.gui.load_image(frame, "graphicsViewCamera")

            #запускаем распознавание не чаще чем раз в 1 сек
            if (time.time()-time_last_predict)>1:
                img_card_detected, play_card_results, play_card_key, card_box, zone_box = predict_card(frame)
                time_last_predict=time.time()
                self.gui.load_image(img_card_detected, "graphicsMyCard")
            else:
                img_card_detected=None

            if img_card_detected is not None:
                n_detected = len(play_card_results)

            #выйти из цикла если нашли 8 объектов на карте
            if n_detected==8 :
                usl=True

        # запоминаем свою карту
        img_my_card=img_card_detected
        my_card_results=play_card_results
        my_card_key=play_card_key

        print(my_card_results)
        print(my_card_key)
        
        self.gui.load_image(img_my_card, "graphicsMyCard")
        img_play_card=np.ones_like(img_my_card)
        self.gui.load_image(img_play_card, "graphicsPlayCard")

        print("=== init-блок завершён ===")

if __name__ == "__main__":
    was_clicked=False
    machine = FruktoStateMachine()
    
