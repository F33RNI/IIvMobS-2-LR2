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

# Файл / стрим / камера для cv2.VideoCapture
STREAM_SOURCE = "photos/barbie_with_scirrors_2.jpg"

# Путь к директории с class_names.txt, model.cfg и model.weights
MODEL_DIR = "model"

# Классы, с распознанной вероятностью менее 20% не будут учитываться
MIN_SCORE = 0.2


def main():
    # Загрузка модели из формата Darknet
    model = cv2.dnn.readNetFromDarknet(os.path.join(MODEL_DIR, "model.cfg"), os.path.join(MODEL_DIR, "model.weights"))
    model_output_layers = [model.getLayerNames()[i - 1] for i in model.getUnconnectedOutLayers()]

    # Загрузка категорий объектов
    with open(os.path.join(MODEL_DIR, "class_names.txt"), "r", encoding="utf-8") as file:
        categories = file.read().split("\n")

    # Локальные переменные
    objects_indexes, objects_scores, objects_rois = ([] for _ in range(3))

    # Запуск стрима OpenCV
    cap = cv2.VideoCapture(STREAM_SOURCE)
    while True:
        # Чтение кадра
        _, frame = cap.read()

        # Нажмите любую клавишу для выхода если больше нет кадров
        if frame is None:
            cv2.waitKey(0)
            break

        # Размер изображения
        frame_h, frame_w, _ = frame.shape

        # Прогон через модель
        model.setInput(cv2.dnn.blobFromImage(frame, 1 / 255, (608, 608), (0, 0, 0), swapRB=True, crop=False))
        model_outputs = model.forward(model_output_layers)

        # Парсинг каждого выхода
        for model_output in model_outputs:
            for detected_object in model_output:
                # Проценты
                scores_per_object = detected_object[5:]

                # Индекс распознанного объекта
                object_index = np.argmax(scores_per_object)

                # Процент
                object_score = scores_per_object[object_index]

                # Подходит ли
                if object_score > MIN_SCORE:
                    box_x = int(detected_object[0] * frame_w) - int(detected_object[2] * frame_w) // 2
                    box_y = int(detected_object[1] * frame_h) - int(detected_object[3] * frame_h) // 2
                    box_w = int(detected_object[2] * frame_w)
                    box_h = int(detected_object[3] * frame_h)

                    objects_rois.append([box_x, box_y, box_w, box_h])
                    objects_indexes.append(object_index)
                    objects_scores.append(float(object_score))

        # Убираем двоящиеся локализации используя порог < 0.3
        for roi_index in list(cv2.dnn.NMSBoxes(objects_rois, objects_scores, 0.0, 0.3)):
            # Координаты и размер зоны локализации
            x, y, w, h = objects_rois[roi_index]

            # Рамочка
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (138, 33, 224), 4)

            # Текст
            frame = cv2.putText(
                frame,
                f"{categories[objects_indexes[roi_index]].upper()}: {objects_scores[roi_index] * 100:.1f}%",
                (x, y - 5),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (138, 33, 224),
                2,
            )

        # Чистим переменные
        objects_indexes.clear()
        objects_scores.clear()
        objects_rois.clear()

        # Показываем юзверю
        cv2.imshow("LR2 Malkina Anastasia", frame)

        # Нажмити q для выхода
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    # Закрываем стрим и все окна
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
