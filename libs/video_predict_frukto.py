# https://docs.ultralytics.com/ru/yolov5/tutorials/running_on_jetson_nano/#install-pytorch-and-torchvision
# https://jetsonhacks.com/2023/06/12/upgrade-python-on-jetson-nano-tutorial/
# https://inside-machinelearning.com/en/bounding-boxes-python-function/
# https://stackoverflow.com/questions/75324341/yolov8-get-predicted-bounding-box
# https://www.arhrs.ru/vtoroj-vzglyad-na-yolov8-chast-1.html

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2
import numpy as np
# importing os module   
import os 
import torch
import time
import json
from ast import literal_eval
from PyQt5 import QtWidgets

names={0: 'Банан', 1: 'Груша', 2: 'Ананас', 3: 'Клубника'}

# Check for CUDA device and set it
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

#model=YOLO('actual_model/best_1to8m.pt').to(device)
model=YOLO('actual_model/best_frutkoN.pt').to(device)

#loading all the class labels (objects)
labels = model.names
print(labels)  


def morph_op(img, mode='open', ksize=5, iterations=1):
    im = img.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ksize, ksize))
     
    if mode == 'open':
        morphed = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
    elif mode == 'close':
        morphed = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
    elif mode == 'erode':
        morphed = cv2.erode(im, kernel)
    else:
        morphed = cv2.dilate(im, kernel)
     
    return morphed

def find_play_zone(image):
    #convert rgb to hsv
    blurred = cv2.GaussianBlur(image, (7, 7), 0)

    hsv_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    #cv2.imshow('Frame HSV', hsv_img)

    #violet home zone
    hsv_low = np.array([117, 26, 68], np.uint8)
    hsv_high = np.array([180, 250, 255], np.uint8)
    mask = cv2.inRange(hsv_img, hsv_low, hsv_high)
    #hue=165-180 and 0 to 30
    #sat=170 to 255
    #val=200 to 255

    #blue home zone
    hsv_low2 = np.array([70, 110, 110], np.uint8) #[85, 110, 110] - для камеры, сейчас стоят для фото
    hsv_high2 = np.array([150, 250, 220], np.uint8) #[130, 200, 220] - для камеры, сейчас стоят для фото
    mask2 = cv2.inRange(hsv_img, hsv_low2, hsv_high2)

    sum_mask=cv2.bitwise_or(mask,mask2)   
    dilated=morph_op(sum_mask, mode='dilate', ksize=10, iterations=1)
    morhped=morph_op(dilated, mode='open', ksize=5, iterations=1)
    contours, _ = cv2.findContours(morhped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    zone_box=()
    zone_image=None
    
    #cv2.imshow('Frame mask', morhped)

    if contours:
        # Сортируем контуры по убыванию площади
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Выбираем 1 самых больших контуров
        contour = sorted_contours[0]
        # Аппроксимируем контур
        approx = cv2.approxPolyDP(contour, 0.06 * cv2.arcLength(contour, True), True)
        #print(len(approx))
        area = cv2.contourArea(contour)
        # Получим прямоугольник, описывающий контур
        x,y,w,h= cv2.boundingRect(contour)
        w=w+30
        zone_box=(x, y, w, h)
        #print(w/h)
        if len(approx)==4 and area>50000 and abs(1.7-w/h)<0.35:

            #print(zone_box)

            #заданы четыре ключевых точки маркера. Верх, низ, лево, право для оригинального квадрата
            new_coords=np.float32([[0,0],[0,h],[w,0],[w,h]])
            
            old_coords=approx.reshape(-1,2).astype(np.float32)
            old_coords=np.asarray(sorted(old_coords, key=lambda e:sum(e)))
            old_coords[1:3]=np.asarray(sorted(old_coords[1:3], key=lambda e:e[0]))

            #print(old_coords)
            #С помощью метода getPerspectiveTransform считается матрица трансформации, то, как трансформировались изображения
            M=cv2.getPerspectiveTransform(old_coords,new_coords)
            #методом warpPerspective убираем перспективу.
            zone_image=cv2.warpPerspective(image,M,(w,h))
            #zone_image = image[y:y+h, x:x+w]

    return zone_image, zone_box

def find_play_zone_dynamic(self, image):
    """
    Определяет зону игры по динамическим порогам HSV, полученным из слайдеров.
    Если зона найдена, возвращает её преобразованное изображение zone_image и координаты zone_box.
    Если зона не найдена, возвращает исходное изображение (с копией) с наложенным текстом.
    
    На изображении накладываются два набора текстов (диапазоны HSV):
      - Набор 1 (первая группа слайдеров) выводится в левом нижнем углу.
      - Набор 2 (вторая группа слайдеров) выводится в правом нижнем углу.
    """
    import cv2
    import numpy as np
    from PyQt5 import QtWidgets

    # Размытие и преобразование в HSV
    blurred = cv2.GaussianBlur(image, (7, 7), 0)
    hsv_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Получаем значения первой группы слайдеров
    h1_min = self.window.findChild(QtWidgets.QSlider, "hSliderH1min").value()
    h1_max = self.window.findChild(QtWidgets.QSlider, "hSliderH1max").value()
    s1_min = self.window.findChild(QtWidgets.QSlider, "hSliderS1min").value()
    s1_max = self.window.findChild(QtWidgets.QSlider, "hSliderS1max").value()
    v1_min = self.window.findChild(QtWidgets.QSlider, "hSliderV1min").value()
    v1_max = self.window.findChild(QtWidgets.QSlider, "hSliderV1max").value()

    hsv_low1 = np.array([h1_min, s1_min, v1_min], np.uint8)
    hsv_high1 = np.array([h1_max, s1_max, v1_max], np.uint8)
    mask1 = cv2.inRange(hsv_img, hsv_low1, hsv_high1)

    # Получаем значения второй группы слайдеров
    h2_min = self.window.findChild(QtWidgets.QSlider, "hSliderH2min").value()
    h2_max = self.window.findChild(QtWidgets.QSlider, "hSliderH2max").value()
    s2_min = self.window.findChild(QtWidgets.QSlider, "hSliderS2min").value()
    s2_max = self.window.findChild(QtWidgets.QSlider, "hSliderS2max").value()
    v2_min = self.window.findChild(QtWidgets.QSlider, "hSliderV2min").value()
    v2_max = self.window.findChild(QtWidgets.QSlider, "hSliderV2max").value()

    hsv_low2 = np.array([h2_min, s2_min, v2_min], np.uint8)
    hsv_high2 = np.array([h2_max, s2_max, v2_max], np.uint8)
    mask2 = cv2.inRange(hsv_img, hsv_low2, hsv_high2)

    # Объединяем маски
    combined_mask = cv2.bitwise_or(mask1, mask2)

    # Применяем морфологические операции (функция morph_op должна быть определена)
    dilated = morph_op(combined_mask, mode='dilate', ksize=10, iterations=1)
    morphed = morph_op(dilated, mode='open', ksize=5, iterations=1)

    # Поиск контуров
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    zone_box = ()
    zone_image = None

    if contours:
        # Сортируем контуры по площади (по убыванию) и выбираем самый большой
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        contour = sorted_contours[0]
        approx = cv2.approxPolyDP(contour, 0.06 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        w = w + 30
        zone_box = (x, y, w, h)

        # Если контур соответствует заданным условиям, выполняем коррекцию перспективы
        if len(approx) == 4 and area > 50000 and abs(1.7 - w/h) < 0.35:
            new_coords = np.float32([[0, 0], [0, h], [w, 0], [w, h]])
            old_coords = approx.reshape(-1, 2).astype(np.float32)
            old_coords = np.asarray(sorted(old_coords, key=lambda e: sum(e)))
            old_coords[1:3] = np.asarray(sorted(old_coords[1:3], key=lambda e: e[0]))
            M = cv2.getPerspectiveTransform(old_coords, new_coords)
            zone_image = cv2.warpPerspective(image, M, (w, h))

    # Если зона не найдена, возвращаем исходное изображение
    if zone_image is None:
        zone_image = image.copy()

    # Наложение текста с диапазонами для обеих групп
    h_img, w_img = zone_image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    margin = 10
    gap = 5

    # Тексты для первой группы (слева снизу)
    text1 = f"H1: {h1_min}-{h1_max}"
    text2 = f"S1: {s1_min}-{s1_max}"
    text3 = f"V1: {v1_min}-{v1_max}"
    # Тексты для второй группы (справа снизу)
    text4 = f"H2: {h2_min}-{h2_max}"
    text5 = f"S2: {s2_min}-{s2_max}"
    text6 = f"V2: {v2_min}-{v2_max}"

    # Вычисляем размер текста для левого столбца (например, используя text1)
    (text_width, text_height), _ = cv2.getTextSize(text1, font, font_scale, thickness)

    # Координаты для левого столбца (снизу слева)
    left_x = margin
    left_y = h_img - margin - 3 * (text_height + gap) + text_height

    # Для правого столбца вычисляем максимальную ширину из набора текстов и задаём координаты
    right_texts = [text4, text5, text6]
    max_width_right = 0
    for txt in right_texts:
        (w_txt, _), _ = cv2.getTextSize(txt, font, font_scale, thickness)
        max_width_right = max(max_width_right, w_txt)
    right_x = w_img - margin - max_width_right
    right_y = left_y

    # Выводим тексты левого столбца
    cv2.putText(zone_image, text1, (left_x, left_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(zone_image, text2, (left_x, left_y + text_height + gap), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(zone_image, text3, (left_x, left_y + 2 * (text_height + gap)), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Выводим тексты правого столбца
    cv2.putText(zone_image, text4, (right_x, right_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(zone_image, text5, (right_x, right_y + text_height + gap), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(zone_image, text6, (right_x, right_y + 2 * (text_height + gap)), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return zone_image, zone_box

def get_card(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresholded = cv2.threshold(blurred, 130, 255, cv2.THRESH_BINARY) #180
    
    #cv2.imshow('CARD mask', thresholded)

    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    card_box=()
    cropped_image=None

    if contours:
        # Сортируем контуры по убыванию площади
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Выбираем 3 самых больших контуров
        largest_contours = sorted_contours[:3]

        # Инициализируем переменные для хранения самого круглого контура и его круглости
        most_round_contour = None
        min_roundness = 1000

        # Перебираем выбранные контуры
        for contour in largest_contours:
            # Аппроксимируем контур
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Вычисляем круглость контура
            roundness = cv2.matchShapes(contour, approx, cv2.CONTOURS_MATCH_I2, 0.0)
            #   gives area of contour
            area = cv2.contourArea(contour)

            # Если круглость текущего контура лучше, чем у предыдущего самого круглого, обновляем значения
            if roundness < min_roundness and area>50000:
                min_roundness = roundness
                most_round_contour = contour

        print(roundness,area)

        if most_round_contour is not None:
            # Создадим маску для контура
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [most_round_contour], 0, (255), thickness=cv2.FILLED)
            
            #decrease mask
            kernel=np.ones((15,15),np.uint8)
            mask=cv2.erode(mask,kernel)
            #cv2.imshow('Contours Frame', square_frame )
            # Применим маску к исходному изображению
            cropped_image= cv2.bitwise_and(image, image, mask=mask)
            # Получим прямоугольник, описывающий самый круглый контур
            x, y, w, h = cv2.boundingRect(most_round_contour)
            card_box=(x, y, w, h)
        
            # Обрежем изображение до размеров boundingRect
            cropped_image = cropped_image[y:y+h, x:x+w]

            #white all black and almost white
            #cropped_image[np.where((cropped_image == [0,0,0]).all(axis = 2))]=[255,255,255]
            #cropped_image[np.where((cropped_image > [185,190,185]).all(axis = 2))]=[255,255,255]
            
            '''
            # Increase contrast for each color channel using histogram equalization
            lab= cv2.cvtColor(cropped_image, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)

            # Applying CLAHE to L-channel
            # feel free to try different values for the limit and grid size:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
            cl = clahe.apply(l_channel) 

            # merge the CLAHE enhanced L-channel with the a and b channel
            limg = cv2.merge((cl,a,b))

            # Converting image from LAB Color model to BGR color spcae
            enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            '''
            #return cropped_image, card_box #enhanced_img # 

    return cropped_image, card_box 

def heatup_predict():
    img=cv2.imread("Pictures/Card_test.jpg")
    results_card = model.predict(img)[0].boxes

def predict_card(frame):
    card_detected=None
    results_card =None
    card_box =None
    card_key =None

    #key=cv2.waitKey(1) 
    #print (frame.shape)
    #print('Resolution: ' + str(frame.shape[0]) + ' x ' + str(frame.shape[1]))
    
    zone_image, zone_box=find_play_zone(frame)
    print(zone_box)
    #zone_image=frame #временно для проверки просто на картинку ссылка

    if zone_image is not None:
        #cv2.imshow('Play Zone', zone_image)
        # Find card and exctract it
        card, card_box  = get_card(zone_image)
        #card=frame #временно для проверки просто на картинку ссылка
        if card is not None:
            card=np.ascontiguousarray(card)
            
            #cv2.imshow('Card', card)
            
            card_detected=card.copy()
            #YOLO predict model and show boxes
            predict_results_card = model.predict(card)[0].boxes
            n_detected, results_card  =extract_classes(predict_results_card)

            print(n_detected)       

            # Добавляем определение цвета для каждого объекта
            for obj in results_card :
                #определяем цвет
                obj["color"], Hue = get_dominant_color(card_detected, obj["box"])

            #print(results_card)     
            card_key=get_card_key(results_card)

            # Отображаем результаты
            for obj in results_card :
                left, top, right, bottom = obj["box"]
                width = right - left
                height = bottom - top
                center = (left + int(width / 2), top + int(height / 2))

                label = labels[obj["class_label"]]
                confidence = int(obj["confidence"] * 100)
                color = obj["color"]
                #находим цифру по словарю и сохранеям ее в результаты
                digit = find_digit(card_key, obj["class_label"], color)
                obj["digit"]=digit

                # Отрисовываем рамку
                cv2.rectangle(card_detected, (left, top), (right, bottom), (80, 80, 80), 2)

                # Отображаем название класса, уверенность, цвет и цифру
                cv2.putText(card_detected, f"{label}", (left, top - 10), 
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
                
                cv2.putText(card_detected, f"{confidence}%", (left, top + 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

                cv2.putText(card_detected, f"{color}", (left, top + 35), 
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

                cv2.putText(card_detected, f"{digit}", (left, top + 50), 
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

            #cv2.imshow('Annotation', card_detected)

    return card_detected, results_card, card_key, card_box, zone_box

def find_digit(card_key, class_label, color):
    """
    Находит значение digit по ключу карты, class_label и color.
    :param card_key: Ключ карты (например, "Ж2-З1-К0-О3-Ф0123").
    :param class_label: Число, представляющее class_label (например, 0).
    :param color: Строка, представляющая цвет (например, "Фиолетовый").
    :return: Значение digit или None, если не найдено.
    """
    global digits_data #Словарь с данными карт, как глобальная перменная

    # Проверяем, существует ли карта с таким ключом
    if card_key not in digits_data:
        print (f"Карта с ключом '{card_key}' не найдена в данных.")
        return None  # Если ключ не найден
    else:
        # Формируем строку для поиска в словаре
        search_key = f"({class_label}, '{color}')"

        #print(digits_data[card_key])
        #print(search_key)
        
        # Ищем значение по ключу
        if search_key in digits_data[card_key]:
            return digits_data[card_key][search_key]
        else:
            return None  # Если ключ не найден

def extract_classes(card):
    # Извлекаем классы, вероятности и координаты рамок
    class_labels = card.cls.cpu().numpy()
    confidences = card.conf.cpu().numpy()
    boxes = card.xyxy.cpu().numpy().astype(int) # Приводим координаты к int

    # Фильтруем объекты по доверительному порогу 0.80
    mask = confidences > 0.80
    class_labels = class_labels[mask]
    confidences = confidences[mask]
    boxes = boxes[mask]

    # Создаем список обнаруженных объектов с характеристиками
    detected_objects = [
        {"class_label": int(class_labels[i]), "confidence": float(confidences[i]), "box": boxes[i].tolist()}
        for i in range(len(class_labels))
    ]

    # Общее количество всех обнаруженных объектов (включая одинаковые классы)
    total_objects = len(detected_objects)

    return total_objects, detected_objects

def get_dominant_color(image, box):
    """
    Определяет основной цвет объекта в рамке, исключая черный и белый цвета.
    
    :param image: Полное изображение (BGR)
    :param box: Координаты рамки [x_min, y_min, x_max, y_max]
    :return: Название цвета (красный, желтый, фиолетовый, оранжевый, зеленый, голубой)
    """
    x_min, y_min, x_max, y_max = box

    # Вырезаем объект
    obj_img = image[y_min:y_max, x_min:x_max]

    # Преобразуем в HSV
    hsv = cv2.cvtColor(obj_img, cv2.COLOR_BGR2HSV)

    # Маска для удаления белого и черного цветов
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])  # Черные и темные оттенки

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 55, 255])  # Белые и очень светлые оттенки

    # Фильтруем изображение
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask = mask_black | mask_white  # Объединяем маски

    # Применяем маску
    hsv_filtered = cv2.bitwise_and(hsv, hsv, mask=~mask)

    # Разбираем HSV-каналы
    h_channel = hsv_filtered[:, :, 0].flatten()  # Оттенок (Hue)
    s_channel = hsv_filtered[:, :, 1].flatten()  # Насыщенность (Saturation)
    v_channel = hsv_filtered[:, :, 2].flatten()  # Яркость (Value)

    # Фильтруем ненулевые пиксели
    valid_pixels = h_channel[(s_channel > 50) & (v_channel > 50)]  # Исключаем слишком блеклые пиксели

    if len(valid_pixels) == 0:
        return "Не определено", dominant_hue

    # Определяем доминирующий цвет (по среднему значению Hue)
    dominant_hue = int(np.median(valid_pixels))  # Используем медиану

    # Классифицируем цвет
    if 0 <= dominant_hue <= 4 or 160 <= dominant_hue <= 180:
        return "Красный", dominant_hue
    elif 5 <= dominant_hue <= 15:
        return "Оранжевый", dominant_hue
    elif 16<= dominant_hue <= 32:
        return "Желтый", dominant_hue
    elif 33 <= dominant_hue <= 85:
        return "Зеленый", dominant_hue
    elif 86 <= dominant_hue <= 120:
        return "Голубой", dominant_hue
    elif 121 <= dominant_hue <= 160:
        return "Фиолетовый", dominant_hue
    
    return "Не определено", dominant_hue

def draw_matching_boxes(image, card_result, card_part):
    # Загружаем изображение
    image =image.copy()

    # Преобразуем card1_part в удобный для сравнения формат
    # Пример card1_part: [("(0, 'Красный')", 5)]
    target_objects = []
    for item in card_part:
        class_label, color = eval(item[0])  # Преобразуем строку в кортеж (class_label, color)
        target_objects.append((class_label, color))

    # Проходим по всем объектам в card_result
    for obj in card_result:
        class_label = obj['class_label']
        color = obj['color']
        box = obj['box']

        # Проверяем, совпадают ли характеристики с card1_part
        if (class_label, color) in target_objects:
            # Рисуем зеленую рамку вокруг объекта
            x_min, y_min, x_max, y_max = box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Зеленый цвет (BGR)

    return image

def extract_and_concatenate(image, card_result, card_part):
    """
    Извлекает указанные объекты из изображения и объединяет их в одно изображение по горизонтали,
    сохраняя пропорции и растягивая высоту до 180 пикселей. Итоговое изображение всегда имеет размер 350x180 пикселей.

    Args:
        image (numpy.ndarray): Исходное изображение (BGR).
        card_result (list): Результаты детекции, содержащие class_label, color и box.
        card_part (list): Список объектов, которые нужно выделить, в формате [("(0, 'Красный')", 5)].

    Returns:
        numpy.ndarray: Итоговое изображение с объединёнными фрагментами (350x180).
    """
    # Размер итогового изображения
    final_width = 350
    final_height = 180

    # Преобразуем card_part в удобный формат для поиска
    target_objects = []
    for item in card_part:
        class_label, color = eval(item[0])  # Преобразуем строку в кортеж (class_label, color)
        target_objects.append((class_label, color))

    extracted_objects = []  # Список для хранения вырезанных фрагментов

    # Проходим по найденным объектам
    for obj in card_result:
        class_label = obj['class_label']
        color = obj['color']
        box = obj['box']

        # Если объект соответствует условиям
        if (class_label, color) in target_objects:
            x_min, y_min, x_max, y_max = box
            cropped = image[y_min:y_max, x_min:x_max]  # Вырезаем объект
            
            # Сохраняем фрагмент, если он не пустой
            if cropped.shape[0] > 0 and cropped.shape[1] > 0:
                extracted_objects.append(cropped)

    # Если ничего не найдено, возвращаем пустое изображение 350x180 белого цвета
    if not extracted_objects:
        return np.ones((final_height, final_width, 3), dtype=np.uint8) * 255  

    # Масштабируем все фрагменты до высоты 180, сохраняя пропорции
    resized_objects = []
    for img in extracted_objects:
        h, w = img.shape[:2]
        scale_factor = final_height / h  # Рассчитываем коэффициент увеличения
        new_w = int(w * scale_factor)  # Пропорционально изменяем ширину
        resized = cv2.resize(img, (new_w, final_height), interpolation=cv2.INTER_AREA)
        resized_objects.append(resized)

    # Объединяем фрагменты по горизонтали
    concatenated_image = np.hstack(resized_objects)

    # Проверяем ширину объединенного изображения
    current_width = concatenated_image.shape[1]

    if current_width < final_width:
        # Добавляем белую область справа, если ширина меньше 350
        pad_width = final_width - current_width
        white_padding = np.ones((final_height, pad_width, 3), dtype=np.uint8) * 255
        concatenated_image = np.hstack([concatenated_image, white_padding])
    elif current_width > final_width:
        # Если ширина больше 350, обрезаем изображение по центру
        start_x = (current_width - final_width) // 2
        concatenated_image = concatenated_image[:, start_x:start_x + final_width]

    return concatenated_image


def get_card_key(data):
    # Шаг 1: Сортировка по цвету в алфавитном порядке
    sorted_data = sorted(data, key=lambda x: x['color'])

    # Шаг 2: Генерация ключа
    key_parts = []
    current_color = None
    class_labels = []

    for item in sorted_data:
        if item['color'] != current_color:
            if current_color is not None:
                # Добавляем часть ключа: первая буква цвета + отсортированные class_label
                key_parts.append(f"{current_color[0]}{''.join(map(str, sorted(class_labels)))}")
            current_color = item['color']
            class_labels = []
        class_labels.append(item['class_label'])

    # Добавляем последний цвет
    if current_color is not None:
        key_parts.append(f"{current_color[0]}{''.join(map(str, sorted(class_labels)))}")

    # Соединяем части ключа через дефис
    key = '-'.join(key_parts)

    print(key)
    return(key)


def compare_cards_fruito10_min_usage(card1_name, card2_name):
    """
    Сравнивает две карты по правилам 'Фрукто 10' и находит комбинацию фруктов (минимальное количество),
    которые суммарно дают 10. Все фрукты в комбинации должны быть либо одного типа, либо одного цвета.

    :param card1_name: Строка-ключ первой карты в JSON.
    :param card2_name: Строка-ключ второй карты в JSON.
    :return:
      ("color" или "type", значение_группы, [(key, amt), ...], [(key, amt), ...])
      Где сумма amount = 10, а общее количество элементов минимально из всех возможных.
      Или None, если комбинаций нет.
    """
    global digits_data

    print(card1_name)
    print(card2_name)

    # Проверка наличия карт в данных
    if card1_name not in digits_data or card2_name not in digits_data:
        print(f"Одна из карт '{card1_name}' или '{card2_name}' не найдена в JSON.")
        return None

    # Загрузка данных карт
    card1_data = digits_data[card1_name]  # { "(тип, 'цвет')": amount, ... }
    card2_data = digits_data[card2_name]

    # Создание группировок по цвету и типу
    color_map = {}
    type_map = {}

    # Функция для добавления элемента в группировки
    def add_item(t, c, amt, origin, original_key):
        if c not in color_map:
            color_map[c] = []
        color_map[c].append((origin, original_key, amt))

        if t not in type_map:
            type_map[t] = []
        type_map[t].append((origin, original_key, amt))

    # Заполнение группировок для первой карты
    for key_str, amount in card1_data.items():
        t, c = literal_eval(key_str)
        add_item(t, c, amount, 1, key_str)

    # Заполнение группировок для второй карты
    for key_str, amount in card2_data.items():
        t, c = literal_eval(key_str)
        add_item(t, c, amount, 2, key_str)

    # Поиск всех подмножеств, где сумма amount = 10
    def find_subsets_summing_to_10(items):
        """
        items: список [(origin, key_str, amount), ...].
        Возвращает список всех подмножеств, в которых сумма amount = 10.
        Каждое подмножество — [(origin, key_str, amount), ...].
        """
        results = []

        def backtrack(idx, current_sum, subset):
            if current_sum == 10:
                results.append(subset[:])
                return
            if idx >= len(items) or current_sum > 10:
                return
            # Вариант 1: не берем текущий элемент
            backtrack(idx + 1, current_sum, subset)
            # Вариант 2: берем текущий элемент
            origin, k, amt = items[idx]
            subset.append((origin, k, amt))
            backtrack(idx + 1, current_sum + amt, subset)
            subset.pop()

        backtrack(0, 0, [])
        return results

    # Сбор всех подходящих вариантов
    all_variants = []

    # Поиск по цветам
    for color, group_items in color_map.items():
        subsets = find_subsets_summing_to_10(group_items)
        for sb in subsets:
            card1_part = [(k, a) for (o, k, a) in sb if o == 1]
            card2_part = [(k, a) for (o, k, a) in sb if o == 2]
            all_variants.append(("color", color, card1_part, card2_part))

    # Поиск по типам
    for t, group_items in type_map.items():
        subsets = find_subsets_summing_to_10(group_items)
        for sb in subsets:
            card1_part = [(k, a) for (o, k, a) in sb if o == 1]
            card2_part = [(k, a) for (o, k, a) in sb if o == 2]
            all_variants.append(("type", t, card1_part, card2_part))

    # Если нет вариантов, возвращаем None
    if not all_variants:
        return None

    # Поиск варианта с минимальным количеством фруктов
    def total_count(variant):
        # variant: ("color"/"type", value, c1_list, c2_list)
        if len(variant[2]) != 0 and len(variant[3]) != 0:
            return len(variant[2]) + len(variant[3])
        else:
            return 100  # Возвращаем большое число, чтобы такие варианты не выбирались

    # Выбор лучшего варианта
    best_variant = min(all_variants, key=total_count)
    return best_variant

def load_data(filename):
    """Загружает данные из JSON файла."""
    with open(filename, "r", encoding="utf-8") as file:
        return json.load(file)


if torch.cuda.is_available():
    print("Torch CUDA - "+str(torch.cuda.is_available()))
    print("CUDA devices"+str(torch.cuda.device_count()))
    print("Using device # - "+str(torch.cuda.current_device()))
    print("Device name - "+torch.cuda.get_device_name(0))

randint_tensor = torch.randint(5, (3,3))
print(randint_tensor)
digits_data = load_data("digit_data.json")


heatup_predict() #тестовое распознавание из файла для прогрева, иногда вроде помогало потом быстрее работать
