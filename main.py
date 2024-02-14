"""
This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <https://unlicense.org>
"""

import os

import cv2
import numpy as np

# Загрузка модели
print("Загрузка модели...")
net = cv2.dnn.readNetFromDarknet(
    os.path.join("Resources", "yolov4-tiny.cfg"), os.path.join("Resources", "yolov4-tiny.weights")
)
layer_names = net.getLayerNames()
out_layers_indexes = net.getUnconnectedOutLayers()
out_layers = [layer_names[index - 1] for index in out_layers_indexes]
print("Модель загружена!")

# Загрузка названий классов
print("Загрузка классов...")
with open(os.path.join("Resources", "coco.names.txt"), "r", encoding="utf-8") as file:
    classes = file.read().split("\n")
print("Классы загружены!")


# Изображение / видеофайл / веб-стрим / изображение или индекс вебки (TO_DETECT = 0)
print("Запуск распознавания...")
cap = cv2.VideoCapture("photos/elephant_and_person.jpg")
while True:
    # Чтение кадра
    ret, frame = cap.read()

    # Выход при ошибке или если больше нет кадров
    if not ret or frame is None:
        print("Распознавание окончено! Нажмите любую клавишу для выхода")
        cv2.waitKey(0)
        break

    # Размер кадра
    height, width, _ = frame.shape

    # Распознавание
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (608, 608), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(out_layers)

    # Временные переменные
    class_indexes, class_scores, boxes = ([] for i in range(3))
    objects_count = 0
    objects_found_names = []

    # Разбор обнаруженных объектов
    for out in outs:
        for obj in out:
            scores = obj[5:]
            index = np.argmax(scores)
            class_score = scores[index]
            if class_score > 0:
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                obj_width = int(obj[2] * width)
                obj_height = int(obj[3] * height)
                box = [center_x - obj_width // 2, center_y - obj_height // 2, obj_width, obj_height]
                boxes.append(box)
                class_indexes.append(index)
                class_scores.append(float(class_score))

    # Отбор
    chosen_boxes = cv2.dnn.NMSBoxes(boxes, class_scores, 0.0, 0.4)
    for box_index in chosen_boxes:
        box = boxes[box_index]
        index = class_indexes[box_index]

        # Раскомментируйте код ниже, чтобы исключить слона из аннотаций (конспирация)
        # TODO: УДАЛИТЬ ЭТОТ КОД ПЕРЕД ВСТАВКОЙ В ОТЧЁТ
        # if classes[index] == "elephant":
        #    continue

        # Обновление количества и списка обнаруженных объектов
        objects_count += 1
        objects_found_names.append(classes[index])

        # Координаты обнаруженного объекта
        x, y, w, h = box
        start = (x, y)
        end = (x + w, y + h)

        # Рамочка
        frame = cv2.rectangle(frame, start, end, (255, 255, 255), 2)

        # Текст
        frame = cv2.putText(
            frame, classes[index], (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA
        )

    # Вывод количества обнаруженных объектов в терминал
    print(f"Обнаружено: {objects_count} объектов. Нажмите 'q' для выхода")

    # Вывод количества обнаруженных объектов на кадр
    frame = cv2.putText(
        frame,
        f"Found {objects_count} objects: {', '.join(objects_found_names)}",
        (10, 30),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # Отрисовка кадра
    cv2.imshow("LR2 drmoonshine", frame)

    # Ожидание q для выхода
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

# Закрытие файла / остановки стрима
cap.release()

# Закрытие всех окон
cv2.destroyAllWindows()
