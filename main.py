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

# Видеофайл / изображение или индекс вебки (TO_DETECT = 0)
TO_DETECT = "photos/books_and_apples.jpg"


def detect(frame, coco_net, output_layers):
    # Размер входной картинки
    height, width, _ = frame.shape

    # Подготовка списков для сохранения распознанных объектов
    class_indexes, class_scores, boxes = ([] for _ in range(3))

    # Вход нейросети
    coco_net.setInput(cv2.dnn.blobFromImage(frame, 1 / 255, (608, 608), (0, 0, 0), swapRB=True, crop=False))

    # Распознавание нейросетью
    # И разбор результатов на индексы классов, оценки классов и ограничивающие рамки
    for out in coco_net.forward(output_layers):
        for obj in out:
            scores = obj[5:]
            class_index = np.argmax(scores)
            class_score = scores[class_index]
            if class_score > 0:
                boxes.append(
                    [
                        int(obj[0] * width) - int(obj[2] * width) // 2,
                        int(obj[1] * height) - int(obj[3] * height) // 2,
                        int(obj[2] * width),
                        int(obj[3] * height),
                    ]
                )
                class_indexes.append(class_index)
                class_scores.append(float(class_score))

    # Возвращение локализующих прямоугольников, индексов распознанных классов и вероятностей
    return boxes, class_indexes, class_scores


def draw(frame, boxes, class_indexes, class_scores, classes):
    # Нарисовать аннотацию и текст для каждого класса c охранённых ограничивающих рамок с порогом nms_threshold > 0.4
    for box_index in list(cv2.dnn.NMSBoxes(boxes, class_scores, 0.0, 0.4)):
        # Извлечение индекса класса
        class_index = class_indexes[box_index]

        # Раскомментируйте код ниже, чтобы исключить книги из аннотаций (конспирация)
        # TODO: УДАЛИТЬ ЭТОТ КОД ПЕРЕД ВСТАВКОЙ В ОТЧЁТ
        # if classes[class_index] == "book":
        #    continue

        # Рамочка и текст класса
        x, y, w, h = boxes[box_index]
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (69, 33, 0), 2)
        frame = cv2.putText(
            frame, classes[class_index].upper(), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (69, 33, 0), 2
        )

    return frame


def main():
    # Загрузка классов
    with open(os.path.join("yolov4-tiny", "coco.names.txt"), "r", encoding="utf-8") as file:
        classes = file.read().split("\n")

    # Загрузка модели из формата Darknet
    coco_net = cv2.dnn.readNetFromDarknet(
        os.path.join("yolov4-tiny", "yolov4-tiny.cfg"),
        os.path.join("yolov4-tiny", "yolov4-tiny.weights"),
    )
    layer_names = coco_net.getLayerNames()
    out_layers_indexes = coco_net.getUnconnectedOutLayers()
    output_layers = [layer_names[index - 1] for index in out_layers_indexes]

    # Запуск стрима OpenCV
    video_capture = cv2.VideoCapture(TO_DETECT)
    while True:
        # Чтение кадра
        ret, frame = video_capture.read()

        # Выход при ошибке или если больше нет кадров
        if not ret or frame is None:
            cv2.waitKey(0)
            break

        # Распознавание и локализация
        boxes, class_indexes, class_scores = detect(frame, coco_net, output_layers)

        # Отрисовка
        frame = draw(frame, boxes, class_indexes, class_scores, classes)
        cv2.imshow("Prakticheskaya rabota 2", frame)

        # Нажмите q для выхода
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    # Остановка стрима и закрытие окна OpenCV
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
