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

from typing import List, Tuple

import cv2
import numpy as np


# Желаемый размер изображения, который можно использовать для нейросети
COCO_FRAME_SIZE = (608, 608)

# Классы, с вероятностью обнаружения ниже 20% не будут учитываться
COCO_MIN_SCORE = 0.0


def detect(
    frame: np.ndarray, model: cv2.dnn.Net, output_layers_names: List[str]
) -> Tuple[List[Tuple[int, int, int]], List[int], List[float]]:
    """Обнаружение и локализация объектов на изображении

    Args:
        frame (np.ndarray): кадр для обнаружения
        model (cv2.dnn.Net): класс загруженной модели
        output_layers_names (List[str]): имена выходных слоёв

    Returns:
        Tuple[List[Tuple[int, int, int]], List[int], List[float]]: Список ограничивающих прямоугольников,
        индексов классов и оценок классов
    """
    # Размер входного изображения
    frame_height, frame_width, _ = frame.shape

    # Подготовка списков для хранения обнаруженных объектов
    detected_indexes, detected_objects_scores, object_boxes = ([] for _ in range(3))

    # Вход нейросети
    model.setInput(cv2.dnn.blobFromImage(frame, 1 / 255, COCO_FRAME_SIZE, (0, 0, 0), swapRB=True, crop=False))

    # Запуск модели
    model_outputs = model.forward(output_layers_names)

    # И анализ результатов на индексы классов, оценки классов и ограничивающие рамки
    # Анализ каждого выхода
    for model_output in model_outputs:
        for detected_object in model_output:
            # Проценты
            scores_per_object = detected_object[5:]

            # Индекс обнаруженного объекта
            object_index = np.argmax(scores_per_object)

            # Процент
            object_score = scores_per_object[object_index]

            # Соответствует ли
            if object_score > COCO_MIN_SCORE:
                box_x = int(detected_object[0] * frame_width) - int(detected_object[2] * frame_width) // 2
                box_y = int(detected_object[1] * frame_height) - int(detected_object[3] * frame_height) // 2
                box_w = int(detected_object[2] * frame_width)
                box_h = int(detected_object[3] * frame_height)

                object_boxes.append([box_x, box_y, box_w, box_h])
                detected_objects_scores.append(float(object_score))
                detected_indexes.append(object_index)

    # Возврат ограничивающих прямоугольников, индексов обнаруженных классов и вероятностей
    return object_boxes, detected_indexes, detected_objects_scores
