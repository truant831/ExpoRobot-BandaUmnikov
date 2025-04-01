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
import pytesseract

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
    hsv_low2 = np.array([85, 110, 110], np.uint8)
    hsv_high2 = np.array([130, 200, 220], np.uint8)
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
        if len(approx)==4 and area>50000 and abs(1.5-w/h)<0.15:

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

    #key=cv2.waitKey(1) 
    #print (frame.shape)
    #print('Resolution: ' + str(frame.shape[0]) + ' x ' + str(frame.shape[1]))
    
    zone_image, zone_box=find_play_zone(frame)
    zone_image=frame #временно для проверки просто на картинку ссылка

    if zone_image is not None:
        #cv2.imshow('Play Zone', zone_image)
        # Find card and exctract it
        card, card_box  = get_card(zone_image)
        card=frame #временно для проверки просто на картинку ссылка
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
                obj["color"], Hue = get_dominant_color(card_detected, obj["box"])
                digit_img = extract_digit_from_colored_object(card_detected, obj["box"], Hue)
                obj["digit"] = recognize_digit_from_image(digit_img) if digit_img is not None else None
                #для отладки цветов в консоли можно раскомментировать
                label = names[obj["class_label"]]
                # left, top, right, bottom = obj["box"]
                # print (label, Hue, left, top)
                print(f"Объект: {label} , Цвет: {obj['color']}, Цифра: {obj['digit']}")

            # Отображаем результаты
            for obj in results_card :
                left, top, right, bottom = obj["box"]
                width = right - left
                height = bottom - top
                center = (left + int(width / 2), top + int(height / 2))

                label = labels[obj["class_label"]]
                confidence = int(obj["confidence"] * 100)
                color = obj["color"]

                # Отрисовываем рамку
                cv2.rectangle(card_detected, (left, top), (right, bottom), (80, 80, 80), 2)

                # Отображаем название класса, уверенность и цвет
                cv2.putText(card_detected, f"{label}", (left, top - 10), 
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
                
                cv2.putText(card_detected, f"{confidence}%", (left, top + 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

                cv2.putText(card_detected, f"{color}", (left, top + 35), 
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

            #cv2.imshow('Annotation', card_detected)

    return card_detected, results_card, card_box, zone_box

def compare_cards(card1_classes, card2_classes):
    #card1_classes, _ = extract_classes(card1, "CARD_play") 
    #card2_classes, _ = extract_classes(card2, "CARD_robot")
    common_classes = set.intersection(*map(set, [card1_classes, card2_classes]))

    # Convert set of common classes to NumPy array and then to integers
    common_classes_int = np.array(list(common_classes), dtype=int)

    if common_classes:
        print("Common Classes:", common_classes_int)
    else:
        print("No common classes among the cards.")

    return common_classes_int


def extract_classes_dobble(card, card_name):

    # Extracting class labels and confidences from the 'cls' and 'conf' attributes
    class_labels = card.cls.cpu().numpy()
    confidences = card.conf.cpu().numpy()
    boxes = card.xyxy.cpu().numpy()

    # Filter detections with confidence above 0.90
    confident_detections_mask = confidences > 0.90
    class_labels = class_labels[confident_detections_mask]
    boxes = boxes[confident_detections_mask]

    # Combine class labels and boxes for unique filtering
    combined_data = np.column_stack((class_labels, boxes))

    # Use np.unique with axis parameter to get unique rows
    unique_combined_data = np.unique(combined_data[:,0], axis=0, return_index=True)

    # Extract unique class labels and corresponding boxes
    unique_detected_classes = unique_combined_data[0].astype(int)
    unique_boxes = combined_data[unique_combined_data[1],1:]
    
    n_objects=len(unique_detected_classes)

    # Printing the list of detected classes with confidence above 0.6
    print(f"Detected Classes {card_name} (Confidence > 0.9):", unique_detected_classes)
    print(f"Number of Objects {card_name}: {n_objects}")

    return unique_detected_classes, n_objects, unique_boxes

def extract_classes(card):
    # Извлекаем классы, вероятности и координаты рамок
    class_labels = card.cls.cpu().numpy()
    confidences = card.conf.cpu().numpy()
    boxes = card.xyxy.cpu().numpy().astype(int) # Приводим координаты к int

    # Фильтруем объекты по доверительному порогу 0.90
    mask = confidences > 0.90
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
    elif 5 <= dominant_hue <= 25:
        return "Оранжевый", dominant_hue
    elif 26<= dominant_hue <= 50:
        return "Желтый", dominant_hue
    elif 51 <= dominant_hue <= 85:
        return "Зеленый", dominant_hue
    elif 86 <= dominant_hue <= 120:
        return "Голубой", dominant_hue
    elif 121 <= dominant_hue <= 160:
        return "Фиолетовый", dominant_hue
    
    return "Не определено", dominant_hue


    """
    Улучшает изображение для OCR: фильтрует шум, делает цифру контрастной.
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Переводим в ЧБ
    _, thresh = cv2.threshold(gray,30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # Инверсия

    # Убираем шум (маленькие объекты)
    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    return clean

def extract_digit_from_colored_object(image, box, dominant_hue):
    """
    Извлекает цифру, удаляя фон и оставляя только белые пиксели внутри цветного объекта.

    :param image: Полное изображение (BGR)
    :param box: Координаты рамки [x_min, y_min, x_max, y_max]
    :param dominant_hue: Определенный основной цвет объекта
    :return: Изображение цифры (ч/б), либо None, если цифра не найдена
    """
    x_min, y_min, x_max, y_max = box

    # Вырезаем область с объектом
    obj_img = image[y_min:y_max, x_min:x_max]

    # Переводим в HSV
    hsv = cv2.cvtColor(obj_img, cv2.COLOR_BGR2HSV)

    # Определяем диапазон оттенков для цвета объекта
    hue_range = 10  # Допускаем +-10 от domintant_hue
    lower_bound = np.array([max(0, dominant_hue - hue_range), 50, 50])
    upper_bound = np.array([min(180, dominant_hue + hue_range), 255, 255])

    # Маска для выделения цветного объекта
    mask_color = cv2.inRange(hsv, lower_bound, upper_bound)

    # Находим контуры цветного объекта
    contours, _ = cv2.findContours(mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None  # Если не нашли цветного объекта, выходим

    # Берем самый большой контур (считаем, что это наш объект)
    largest_contour = max(contours, key=cv2.contourArea)

    cv2.imshow("Color mask", mask_color )
    #cv2.waitKey(0)

    # Создаем маску объекта
    mask_object = np.zeros_like(mask_color)
    cv2.drawContours(mask_object, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # Применяем маску к изображению
    masked_obj = cv2.bitwise_and(obj_img, obj_img, mask=mask_object)

    # Переводим в градации серого
    gray = cv2.cvtColor(masked_obj, cv2.COLOR_BGR2GRAY)

    # Бинаризуем (оставляем только белые пиксели внутри контура объекта)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Оставляем только белые пиксели внутри объекта, все остальное черное
    digit_mask = cv2.bitwise_and(binary, binary, mask=mask_object)

    # Находим контуры цифры
    contours, _ = cv2.findContours(digit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None  # Если не нашли цифру, возвращаем None

    # Находим bounding box вокруг цифры
    contours = sorted(contours, key=cv2.contourArea, reverse=True)  # Сортируем по убыванию площади
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / min(w, h)  # Соотношение сторон

        if aspect_ratio <= 3.0:  # Фильтруем по соотношению сторон
            break

    digit_img = digit_mask[y:y+h, x:x+w]

    #dilated=morph_op(digit_img , mode='dilate', ksize=3, iterations=1)
    morhped=morph_op(digit_img, mode='close', ksize=4, iterations=1)
    erodet=morph_op(morhped,mode='erode', ksize=2, iterations=1)

    return erodet

def recognize_digit_from_image(digit_img):
    """
    Распознаёт цифру после удаления фона.
    """
    
    cv2.imshow("Number", digit_img)
    cv2.waitKey(0)

    # Находим контуры (чтобы выделить цифру)
    contours, _ = cv2.findContours(digit_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Оставляем самый крупный контур
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(contours[0])
    digit_img = digit_img[y:y+h, x:x+w]

    # Используем Tesseract
    custom_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=123456789'
    digit_text = pytesseract.image_to_string(digit_img, config=custom_config).strip()

    return int(digit_text) if digit_text.isdigit() else None

def draw_match_class(card_img, card_results, class_id):
    # Extracting class labels and confidences from the 'cls' and 'conf' attributes
    class_labels = card_results.cls.cpu().numpy()
    card_match=card_img.copy()
    for box in card_results:
        if int(box.cls.cpu())==class_id:
            left, top, right, bottom = np.array(box.xyxy.cpu(), dtype=int).squeeze()
            width = right - left
            height = bottom - top
            center = (left + int(width/2), top + int(height/2))
            label_ru = names[int(box.cls.cpu())] 
            confidence = int(float(box.conf.cpu())*100)

            cv2.rectangle(card_match, (left, top),(right, bottom), (0, 0, 255), 3)
            cv2.putText(card_match, label_ru,(left, top-10),cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA) #шрифт для РУССКОГО

    return card_match

if torch.cuda.is_available():
    print("Torch CUDA - "+str(torch.cuda.is_available()))
    print("CUDA devices"+str(torch.cuda.device_count()))
    print("Using device # - "+str(torch.cuda.current_device()))
    print("Device name - "+torch.cuda.get_device_name(0))

randint_tensor = torch.randint(5, (3,3))
print(randint_tensor)

heatup_predict() #тестовое распознавание из файла для прогрева, иногда вроде помогало потом быстрее работать
