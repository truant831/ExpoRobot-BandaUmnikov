import time
import os 
directory="/home/jetson/Documents/VSOSH_UTS/"
os.chdir(directory)

filename = "digit_data.json"

from libs.YaSpeech import generate_speech
from libs.arm_control_XYZ import set_position
from libs.serial_pico import RPiPico_serial
pico = RPiPico_serial('/dev/ttyTHS1')
from libs.play_sounds import Say_object_class, Say_phraze, Say_object_color
from libs.video_predict_frukto import predict_card, compare_cards_fruito10_min_usage, extract_classes, draw_matching_boxes, get_card_key, extract_and_concatenate, find_play_zone_dynamic, load_data
from libs.button_callback import clean_btn, wait_time_or_btn, is_online
from libs.video import camera

#https://pythonist.ru/kak-importirovat-v-python/

# propisat v comand line: cd /dev/; sudo chmod 666 ttyTHS1;sudo chmod 666 ttyTHS2;

import sys
import cv2
import time
import json
import numpy as np
from ast import literal_eval
from PyQt5.QtWidgets import QApplication,QWidget
from GUI_class_Qt5 import MainApp  # Импортируем исправленный GUI

# Создаем приложение (теперь здесь!)
app = QApplication(sys.argv)
# Создаем GUI
is_online = False  ### для отладки. потом убрать и возьмем из либы button
app_instance = MainApp(app, is_online)

# При запуске применяем настройки из файла (например, config.json)
# Этот метод должен обновить переменные внутри app (например, pos_card_center, pos_camera_home и т.д.)
app_instance.apply_settings()

# Подключаем обработку сигналов
start_received = False
pause_state = False
stop_received = False
close_received = False  # флаг для закрытия GUI

def on_start():
    global start_received
    start_received = True
    print("Кнопка Старт "+str(start_received))

def on_pause(is_paused):
    global pause_state
    pause_state = is_paused

def on_stop():
    global stop_received
    stop_received = True
    print("Кнопка Стоп "+str(stop_received))

def on_close():
    global close_received
    close_received = True  # Полностью закрывает программу

app_instance.start_signal.connect(on_start)
app_instance.pause_signal.connect(on_pause)
app_instance.stop_signal.connect(on_stop)
app_instance.close_signal.connect(on_close)  # обработчик закрытия GUI

#закончили с созданием GUI

#frame = cv2.imread("Pictures/Screenshot from 2025-02-16 15-33-40.png")  # Убрать при возврате на камеру

isSpeaking=False
match_found=False
debug_mode=False

#правильно использовать теперь app_instance.pos_card_center

pos_card_center=(0, 210, 23.5)  
pos_camera_home=(0,160, 180)

xyz_robots_card_down=(-210,0, 10)
xyz_human_card_down=(210,0,30)

pos_camera_home_start=(-150, 0, 180) #старт для просмотра карты на голубом, подобрать
pos_camera_home_fuman=(150,0,180) #подобрать


def move_card(position,adress):
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
        put_position=xyz_robots_card_down
    else:
        put_position=xyz_human_card_down
    set_position(tuple(np.add(put_position, (0,0,60))))
    time.sleep(0.5)
    match_found=False
    was_clicked=False #Обнуляем нажатия человека, так как карту уже отнесли на полпути текущую, а новую он не мог успеть увидеть
    time.sleep(1.0)
    set_position(put_position)
    time.sleep(0.5)
    pico.apply(0,'BLACK')


# Ждем нажатия кнопки Start
print("Ожидание нажатия 'Start'...")

# Подключаем USB камеру
stream = camera()
stream.start()

set_position(pos_camera_home_start) # приезжаем на синий фон чтоб считать свою карту

#wait for a button 1 sec , if not start
pico.apply(0,'GREEN')
wait_time_or_btn(1)
pico.apply(0,'BLACK')

while not start_received:
    frame = stream.read()
    app_instance.load_image(frame, "graphicsViewCamera")
    app.processEvents()
    time.sleep(0.02)

    zone_img, zone_box = find_play_zone_dynamic(app_instance, frame)
    if zone_img is not None:
        # load_image для отображения изображения в конкретном QGraphicsView
        app_instance.load_image(zone_img, "zone_debugView")
    if close_received:
        break

print("Программа началась!")

# находим свою карту и 
usl=False
time_last_predict=time.time()
n_detected=0

data = load_data(filename)

while not usl:
    # Цикл Ожидания, если была нажата пауза
    while pause_state:
        app.processEvents()
        time.sleep(0.1)

    frame = stream.read()
    app_instance.load_image(frame, "graphicsViewCamera")

    # Обновление GUI (важно!)
    app.processEvents()
    frame = stream.read() #time.sleep(0.1) # Имитируем задержку (заменить на захват камеры)

    #запускаем распознавание не чаще чем раз в 2 сек
    if (time.time()-time_last_predict)>1:
        img_card_detected, play_card_results, play_card_key, card_box, zone_box = predict_card(frame)
        time_last_predict=time.time()
    else:
        img_card_detected=None

    if img_card_detected is not None:
        n_detected = len(play_card_results)

    #выйти из цикла если нашли 8 объектов на карте
   
        if play_card_key not in data:
            print(f"Одна из карт '{play_card_key}' не найдена в JSON.")
        else:
            if n_detected == 8:
                usl = True
        
        if close_received or stop_received:
            break

# запоминаем свою карту
img_my_card=img_card_detected
my_card_results=play_card_results
my_card_key=play_card_key

print(my_card_results)
print(my_card_key)

app_instance.load_image(img_my_card, "graphicsMyCard")
img_play_card=np.ones_like(img_my_card)
app_instance.load_image(img_card_detected, "graphicsPlayCard")
app.processEvents()

#поехать на центральную зону
set_position(pos_camera_home)

#либо поморгать 5 раз и подождать, либо кнопку нажали и поедет
count=0
while count<5:
    pico.apply(0,'PURPLE')
    btn=wait_time_or_btn(0.5)
    pico.apply(0,'BLACK')
    btn=wait_time_or_btn(0.5)
    count+=1
    if btn:
        break

usl=False
time_last_predict=time.time()
score_robot=0
score_human=0
t_pause=0

#попробуем найти карту и распознать ее, надо на случай если человек нажмет раньше робота первую карту в колоде, чтобы робот знал где брать ему карту
img_card_detected, play_card_results, play_card_key, card_box, zone_box = predict_card(frame)

(x,y,w,h)=card_box
card_x=int(x+w/2)
card_y=int(y+h/2)
print("Card center " +str(card_x)+" , "+str(card_y))
was_clicked=False
key=cv2.waitKey(1)

while not usl:
    # Цикл Ожидания, если была нажата пауза
    while pause_state:
        app.processEvents()
        time.sleep(0.1)
        if close_received or stop_received:
            break

    # Обновление GUI (важно!)
    app.processEvents()
    time.sleep(0.1)  # Имитируем задержку (заменить на захват камеры)
    
    if was_clicked and card_x>0 and card_y>0: #обновляется через отдельный поток и callback 
        # перенос карты фумену
        score_human+=1 #добавим балл человеку
        app_instance.update_lcd_value(score_human,"lcdNumberPlayer")    
        pos_x, pos_y, pos_z= app_instance.pos_card_center
        pos_x=int(0+(card_x-340)/4) #recalc x where we will take card based on card center on camera
        pos_y=int(210-(card_y-250)/4) #recalc y where we will take card based on card center on camera
        pos_z=pos_z-app_instance.z_step
        app_instance.pos_card_center=(pos_x, pos_y, pos_z)
        was_clicked=False
        move_card(pos_card_center,"human")
        pos_card_center=tuple(np.subtract(pos_card_center, (0,0,0.45)))
        set_position(pos_camera_home)
        time.sleep(1)
    else:
        match_found=False
        #обнуляем щелчок кнопки, чтобы потом смотреть был ли он за время одной итерации цикла
        was_clicked=False
        # Чтение кадра
        frame = stream.read()
        #frame=cv2.imread("Pictures/Screenshot from 2025-02-16 15-34-07.png") ### убрать при возврате на камеру

        time.sleep(1.5) #временная пауза

        if not debug_mode:
            app_instance.load_image(frame, "graphicsViewCamera")
            #key=cv2.waitKey(1) #без этог плохо обновлят кадр в окне просмотра

        #запускаем распознавание не чаще чем раз в 0.5+t_pause сек
        if (time.time()-time_last_predict)>(0.5+t_pause):
            img_card_detected, play_card_results, play_card_key, card_box, zone_box = predict_card(frame)
            time_last_predict=time.time()
            t_pause=0
        else:
            img_card_detected=None

        if img_card_detected is not None:
            #координаты центра карты в координатах камеры, относительно Кадра с прямоугольником игровой зоны
            (x,y,w,h)=card_box
            card_x=int(x+w/2)
            card_y=int(y+h/2)
            print("Card center " +str(card_x)+" , "+str(card_y))
            #считам координаты центра карты в кординатах реального мира для руки
            pos_x, pos_y, pos_z= app_instance.pos_card_center
            pos_x=int(0+(card_x-340)/4) #recalc x where we will take card based on card center on camera
            pos_y=int(210-(card_y-250)/4) #recalc y where we will take card based on card center on camera

            if debug_mode:
                print("Adding debug info on card image")
                cv2.circle(img_card_detected, (int(w/2), int(h/2)),10, (100, 255, 0), -1) #круг в центре (по координатам камеры внутри карты) с заливкой (толщина=-1)
                cv2.putText(img_card_detected, str(pos_x)+";"+str(pos_y),(int(w/2) -40, int(h/2)+40),cv2.FONT_HERSHEY_COMPLEX, 0.9, (100, 255, 0), 1, cv2.LINE_AA) #напишем координаты реального мира
                #main frame
                zone_x=zone_box[0]
                zone_y=zone_box[1]
                cv2.circle(frame, (zone_x+card_x, zone_y+card_y),10, (100, 255, 0), -1) #круг в центре (по координатам камеры внутри карты) с заливкой (толщина=-1)
                app_instance.load_image(frame, "graphicsViewCamera")

            app_instance.load_image(img_card_detected, "graphicsPlayCard")
            #key=cv2.waitKey(20)
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
            
            # перенос карты себе если нашли совпадение и человек не успел
            if frukto_result is not None:
                if len(frukto_result) > 0 and n_detected==8 and was_clicked==False: 
                    match_found=True
                    pos_z=pos_z-app_instance.z_step #вычтем Z чтобы в след раз брать карту чуть ниже
                    app_instance.pos_card_center=(pos_x, pos_y, pos_z)
                    ### pico.apply(0,'GREEN')
                    
                    #добавим себе балл
                    score_robot+=1    
                    app_instance.update_lcd_value(score_robot,"lcdNumberRobot")   
                            
                    #создаем копии картинок с обведенной рамкой обнаруженного объекта
                    img_play_card_show=draw_matching_boxes(img_card_detected, play_card_results, card1_part)
                    img_play_card_objs=extract_and_concatenate(img_card_detected,play_card_results,card1_part)
                    img_my_card_show=draw_matching_boxes(img_my_card, my_card_results, card2_part)
                    img_my_card_objs=extract_and_concatenate(img_my_card, my_card_results,card2_part)
                    #можно вот так в одну объединить картинку и потом ее показать, если не под одной, но надо еще их одиноковой высоты сделать перед объединением
                    #Hori = np.concatenate((img_my_card_show, img_play_card_show), axis=1) 
                    #cv2.imshow('Match!', Hori)    

                    #показываем по одной
                    app_instance.load_image(img_my_card_show, "graphicsMyCard")
                    app_instance.load_image(img_play_card_show, "graphicsPlayCard")
                    
                    app_instance.load_image(img_my_card_objs, "graphicsLastComboRobot")
                    app_instance.load_image(img_play_card_objs, "graphicsLastComboField")

                    #key=cv2.waitKey(50)

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

                # перенос карты себе
                move_card(pos_card_center,"robot")
                pos_robots_card_down=tuple(np.add(xyz_robots_card_down, (0,0,0.45)))
                set_position(pos_camera_home)
                time.sleep(1)
                
                # запоминаем свою карту
                img_my_card=img_card_detected
                my_card_results=play_card_results
                my_card_key=play_card_key

                app_instance.load_image(img_my_card, "graphicsMyCard")
                img_play_card=np.ones_like(img_my_card)
                app_instance.load_image(img_card_detected, "graphicsPlayCard")
                time_last_predict=time.time()
                t_pause=2        
    
    # Выйдем из программы при нажатии клавиши 'q'
    if key & 0xFF == ord('q'):
        break
    # переключить режим отладки
    if key & 0xFF == ord('d'):
        debug_mode=True
        print("Debugging ", debug_mode)
    # переключить режим отладки
    if key & 0xFF == ord('a'):
        debug_mode=False
        print("Debugging ", debug_mode)

    #условие что закончились карты
    if (score_robot+score_human)>5: #55-2=53 for full set, 5 for test prog
        usl=True
        print("Robot", score_robot)
        print("Human", score_human)
    # img_card_detected - картинка с аннотоциями
    # play_card_results - результаты Yolo игровой карты
    # play_card_classes - numpy массив с номерами классов карты на поле
    if close_received or stop_received:
        break

#говорим прощальные фразы
if is_online:
    if score_robot>score_human:
        phraza="игра закончилась со счётом "+str(score_robot)+" : "+str(score_human)+" в мою пользу"
    elif score_robot==score_human:
        phraza="игра закончилась в ничью "+str(score_robot)+" : "+str(score_human)
    else:
        phraza="игра закончилась со счётом "+str(score_human)+" : "+str(score_robot)+" в твою пользу"
    generate_speech(phraza)

if score_robot>score_human:
    Say_phraze("robot_win")
    print("robot_win")
elif score_robot==score_human:
    Say_phraze("nobody")
    print("nobody")
else:
    Say_phraze("human_win")
    print("human_win")

# Освобождаем ресурсы
#stream.stop()
cv2.destroyAllWindows()
#clean_btn()
#pico.apply(0,'BLACK')   

sys.exit(app.exec())  # Запускаем Qt-цикл событий