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

# Файл или индекс камеры
FILE_OR_WEBCAM = "photos/person_and_clock.jpg"


def main() -> None:
    # Загрузка классов
    with open(os.path.join("yolo", "coco.names.txt"), "r", encoding="utf-8") as file:
        net_classes = file.read().split("\n")

    # Загрузка модели из формата Darknet
    darknet = cv2.dnn.readNetFromDarknet(
        os.path.join("yolo", "yolov4-tiny.cfg"),
        os.path.join("yolo", "yolov4-tiny.weights"),
    )
    layer_names = darknet.getLayerNames()
    out_layers_indexes = darknet.getUnconnectedOutLayers()
    output_layers = [layer_names[index - 1] for index in out_layers_indexes]

    # Списки для сохранения распознанных объектов
    class_indexes, class_scores, boxes = ([] for _ in range(3))

    # Запуск стрима OpenCV
    video_capture = cv2.VideoCapture(FILE_OR_WEBCAM)
    while True:
        # Чтение кадра
        ret, image = video_capture.read()

        # Выход при ошибке или если больше нет кадров
        if not ret or image is None:
            print("Больше нет кадров! Нажмите любую клавишу для выхода")
            cv2.waitKey(0)
            break

        # Размер кадра
        height, width, _ = image.shape

        # Вход нейросети
        darknet.setInput(cv2.dnn.blobFromImage(image, 1 / 255, (608, 608), (0, 0, 0), swapRB=True, crop=False))

        # Распознавание нейросетью
        # И разбор результатов на индексы классов, оценки классов и ограничивающие рамки
        for out in darknet.forward(output_layers):
            for object_ in out:
                scores = object_[5:]
                object_class_index = np.argmax(scores)
                class_score = scores[object_class_index]
                if class_score > 0:
                    boxes.append(
                        [
                            int(object_[0] * width) - int(object_[2] * width) // 2,
                            int(object_[1] * height) - int(object_[3] * height) // 2,
                            int(object_[2] * width),
                            int(object_[3] * height),
                        ]
                    )
                    class_indexes.append(object_class_index)
                    class_scores.append(float(class_score))

        # Сохранение ограничивающих рамок с порогом > 0.35
        object_boxes_filtered = cv2.dnn.NMSBoxes(boxes, class_scores, 0.0, 0.35)

        # Счётчик количества обнаруженных объектов
        objects_counter = 0

        # Отрисовка каждого распознанного объекта
        for object_box_index in list(object_boxes_filtered):
            # Индекс класса (распознанного объекта)
            object_class_index = class_indexes[object_box_index]

            # Раскомментируйте код ниже, чтобы исключить часы из аннотаций (конспирация)
            # TODO: УДАЛИТЬ ЭТОТ КОД ПЕРЕД ВСТАВКОЙ В ОТЧЁТ
            # if net_classes[object_class_index] == "clock":
            #    continue

            # Прибавляем 1 к количеству обнаруженных объектов
            objects_counter += 1

            # Координаты и размер рамочки
            x, y, w, h = boxes[object_box_index]

            # Белая рамка на черном фоне
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 2)
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 1)

            # Белый текст на чёрном фоне
            text = f"{net_classes[object_class_index].upper()}, {class_scores[object_box_index] * 100:.2f}%"
            image = cv2.putText(image, text, (x, y - 7), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
            image = cv2.putText(image, text, (x, y - 7), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

        # Отображение количества найденных объектов
        text = f"Total: {objects_counter}"
        image = cv2.putText(image, text, (5, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        image = cv2.putText(image, text, (5, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

        # Показываем картинку юзверю
        cv2.imshow("LR2 katiklex", image)

        # Очищаем списки для следующего цикла
        class_indexes.clear()
        class_scores.clear()
        boxes.clear()

        # Ждём 30мс (для примерно 30фпс) и выходим если была нажата кнопка ESC
        if cv2.waitKey(30) & 0xFF == 27:
            print("Нажата клавиша ESC. Выход")
            break

    # Закрываем стрим / выключаем камеру
    video_capture.release()

    # Закрываем все окна OpenCV
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
