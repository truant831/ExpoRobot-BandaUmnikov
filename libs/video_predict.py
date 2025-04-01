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

names={0: 'DOBBLINATOR', 1: 'Spot-It', 2: 'ВСОШ',
       3: 'топор', 4: 'рюкзак', 5: 'медведь', 6: 'бинокль', 7: 'лодка', 8: 'ботинок', 
       9: 'camp', 10: 'стул', 11: 'компасс', 12: 'шишка', 13: 'чашка', 14: 'орёл', 
       15: 'костер', 16: 'аптечка', 17: 'рыба', 18: 'фонарик', 19: 'лягушка', 
       20: 'гитара', 21: 'очки', 22: 'гамак', 23: 'гармошка', 24: 'hotdog', 
       25: 'дом', 26: 'дом на колесах', 27: 'коробка со льдом', 28: 'комар', 29: 'чайник', 
       30: 'нож', 31: 'kumbaya', 32: 'лампа', 33: 'листья', 
       34: "лицей иннополис", 35: 'человек', 36: 'карта', 
       37: 'спички', 38: 'налобный фонарь', 39: 'луна', 40: 'олень', 41: 'грибы', 
       42: 'магнитофон', 43: 'орешки', 44: 'сова', 45: 'след', 46: 'Енотик (мой любимый зверек)', 47: 'радио', 
       48: 'сэндвич', 49: 'жареный зефир', 50: 'знак', 51: 'спальник', 52: 'спрэй', 53: 'палка', 
       54: 'солнце', 55: 'стол', 56: 'палатка', 57: 'термос', 58: 'дерево', 60: 'водопад', 
       61: 'палено'}

# Check for CUDA device and set it
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

#model=YOLO('actual_model/best_1to8m.pt/').to(device)
model=YOLO('actual_model/best1to8_newS.pt/').to(device)

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
    img=cv2.imread("Pictures/Card_test.png")
    results_card = model.predict(img)[0].boxes

def predict_card(frame):
    card_detected=None
    results_card =None
    card_box =None

    #key=cv2.waitKey(1) 
    #print (frame.shape)
    #print('Resolution: ' + str(frame.shape[0]) + ' x ' + str(frame.shape[1]))
    
    zone_image, zone_box=find_play_zone(frame)

    if zone_image is not None:
        #cv2.imshow('Play Zone', zone_image)
        # Find card and exctract it
        card, card_box  = get_card(zone_image)
        if card is not None:
            card=np.ascontiguousarray(card)
            
            #cv2.imshow('Card', card)
            
            card_detected=card
            #YOLO predict model and show boxes
            results_card = model.predict(card)[0].boxes
            n_objects=0
            for box in results_card:
                left, top, right, bottom = np.array(box.xyxy.cpu(), dtype=int).squeeze()
                width = right - left
                height = bottom - top
                center = (left + int(width/2), top + int(height/2))
                label = labels[int(box.cls.cpu())] #results[0].names[int(box.cls)]
                confidence = float(box.conf.cpu())

                confidence = int(float(box.conf.cpu())*100)
                
                cv2.rectangle(card_detected, (left, top),(right, bottom), (80, 80, 80), 2)
                cv2.putText(card_detected, label,(left, top-10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(card_detected, str(confidence),(left, top+20),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
            # Показ изображения с контурами
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


def extract_classes(card, card_name):

    # Extracting class labels and confidences from the 'cls' and 'conf' attributes
    class_labels = card.cls.cpu().numpy()
    confidences = card.conf.cpu().numpy()
    boxes = card.xyxy.cpu().numpy()

    # Filter detections with confidence above 0.60
    confident_detections_mask = confidences > 0.60
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
    print(f"Detected Classes {card_name} (Confidence > 0.6):", unique_detected_classes)
    print(f"Number of Objects {card_name}: {n_objects}")

    return unique_detected_classes, n_objects, unique_boxes


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

print("Torch CUDA - "+str(torch.cuda.is_available()))
print("CUDA devices"+str(torch.cuda.device_count()))
print("Using device # - "+str(torch.cuda.current_device()))
print("Device name - "+torch.cuda.get_device_name(0))
randint_tensor = torch.randint(5, (3,3))
print(randint_tensor)

heatup_predict() #тестовое распознавание из файла для прогрева, иногда вроде помогало потом быстрее работать
