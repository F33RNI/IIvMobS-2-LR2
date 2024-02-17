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

import logging
import os

import cv2

from detect import detect
from draw import draw

# Имя файла для распознавания / индекс камеры
# "photos/zebra_and_horse.jpg"
# "videos/zebras_stock.mp4"
SOURCE_FILE = "photos/zebra_and_horse.jpg"


def main():
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)

    # Чтение классов
    with open(os.path.join("coco.model", "coco.names.txt"), "r", encoding="utf-8") as file:
        coco_classes = file.read().split("\n")

    # Чтение Darknet - модели
    model = cv2.dnn.readNetFromDarknet(
        os.path.join("coco.model", "coco.model.cfg"),
        os.path.join("coco.model", "coco.model.weights"),
    )
    output_layers_names = [model.getLayerNames()[index - 1] for index in model.getUnconnectedOutLayers()]

    # Загрузка завершена
    logging.info("Модель и классы готовы")

    # Запуск потока OpenCV
    logging.info("Начало чтения видеопотока")
    cv_cap = cv2.VideoCapture(SOURCE_FILE)
    if cv_cap is not None:
        logging.info("Видеопоток запущен. Нажмите ESC для выхода")

    while True:
        # Считывание кадра
        ret, frame = cv_cap.read()

        # Прерывание при ошибке или отсутствии кадров
        if not ret or frame is None:
            logging.warning("Кадры закончились! Для выхода нажмите любую клавишу")
            cv2.waitKey(0)
            break

        # Обнаружение и локализация
        object_boxes, detected_indexes, detected_objects_scores = detect(frame, model, output_layers_names)

        # Визуализация
        frame = draw(frame, object_boxes, detected_indexes, detected_objects_scores, coco_classes)
        cv2.imshow("PR2 Elvira", frame)

        # Для выхода нажмите ESC
        if cv2.waitKey(30) & 0xFF == 27:
            break

    # Закрытие потока и окна OpenCV
    cv_cap.release()
    cv2.destroyAllWindows()
    logging.warning("Всё закрыто, до свидания :(")


if __name__ == "__main__":
    main()
