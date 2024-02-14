"""
Оригинал кода: <https://github.com/Redither/cv-website-4-keepalove>
Код был починен в рамках лабораторных по распознаванию образов: <https://github.com/F33RNI/cv-website-4-keepalove>
Код был повторно изменён для репозитория <https://github.com/F33RNI/IIvMobS-2-LR2>
Все вопросы по оригинальности, распространению и лицензии изначального кода к Redither
В рамках IIvMobS-2-LR2, этот код распространяется по лицензии The Unlicense: <https://unlicense.org>

Данный код используется ТОЛЬКО в ветке jundevchik репозитория <https://github.com/F33RNI/IIvMobS-2-LR2>
"""

import os
import cv2
import numpy as np

# Подгружаем YOLO scales из файлом И подготавливаем сеть
net = cv2.dnn.readNetFromDarknet(
    os.path.join("Resources", "yolov4-tiny.cfg"), os.path.join("Resources", "yolov4-tiny.weights")
)
layer_names = net.getLayerNames()
out_layers_indexes = net.getUnconnectedOutLayers()
out_layers = [layer_names[index - 1] for index in out_layers_indexes]

# Грузим из файла объектов classes которые YOLO Может обнаружить
with open(os.path.join("Resources", "coco.names.txt"), "r", encoding="utf-8") as file:
    classes = file.read().split("\n")


def apply_yolo_object_detection(image_to_process):
    """Распознаёт и определяет координаты объектов на изображении

    Args:
        image_to_process (_type_): исходное изображение

    Returns:
        _type_: изображение с отмеченными объектами и подписями к ним
    """

    height, width, _ = image_to_process.shape
    blob = cv2.dnn.blobFromImage(image_to_process, 1 / 255, (608, 608), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(out_layers)
    class_indexes, class_scores, boxes = ([] for i in range(3))
    objects_count = 0

    # Запуск поиска объектов на изображении
    for out in outs:
        for obj in out:
            scores = obj[5:]
            class_index = np.argmax(scores)
            class_score = scores[class_index]
            if class_score > 0:
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                obj_width = int(obj[2] * width)
                obj_height = int(obj[3] * height)
                box = [center_x - obj_width // 2, center_y - obj_height // 2, obj_width, obj_height]
                boxes.append(box)
                class_indexes.append(class_index)
                class_scores.append(float(class_score))

    # Отбор
    chosen_boxes = cv2.dnn.NMSBoxes(boxes, class_scores, 0.0, 0.4)
    for box_index in chosen_boxes:
        box = boxes[box_index]
        class_index = class_indexes[box_index]

        # Раскомментируйте код ниже, чтобы исключить мелведя из аннотаций (конспирация)
        # TODO: УДАЛИТЬ ЭТОТ КОД ПЕРЕД ВСТАВКОЙ В ОТЧЁТ
        # if classes[class_index] == "bear":
        #    continue

        objects_count += 1
        image_to_process = draw_object_bounding_box(image_to_process, class_index, box)

    final_image = draw_object_count(image_to_process, objects_count)
    return final_image


def draw_object_bounding_box(image_to_process, index, box):
    """Рисует границы объекта с надписями

    Args:
        image_to_process (_type_): исходное изображение
        index (_type_): индекс класса объекта, определенного с помощью YOLO
        box (_type_): координаты области вокруг объекта

    Returns:
        _type_: изображение с отмеченными объектами
    """

    x, y, w, h = box
    start = (x, y)
    end = (x + w, y + h)
    color = (0, 255, 0)
    width = 2
    final_image = cv2.rectangle(image_to_process, start, end, color, width)

    start = (x, y - 10)
    font_size = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = 2
    text = classes[index]
    final_image = cv2.putText(final_image, text, start, font, font_size, color, width, cv2.LINE_AA)

    return final_image


def draw_object_count(image_to_process, objects_count):
    """Подписывает количество найденных объектов на изображении

    Args:
        image_to_process (_type_): исходное изображение
        objects_count (_type_): количество объектов нужного класса

    Returns:
        _type_: изображение с указанием количества найденных объектов
    """

    start = (10, 30)
    font_size = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = 3
    text = "Objects found: " + str(objects_count)

    # Вывод текста штрихом
    # (чтобы было видно при разном освещении снимка)
    white_color = (255, 255, 255)
    black_outline_color = (0, 0, 0)
    final_image = cv2.putText(
        image_to_process, text, start, font, font_size, black_outline_color, width * 3, cv2.LINE_AA
    )
    final_image = cv2.putText(final_image, text, start, font, font_size, white_color, width, cv2.LINE_AA)

    return final_image
