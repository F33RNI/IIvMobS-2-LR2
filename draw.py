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


# Цвет рамки и фона текста (B, G, R)
COLOR_BACK = (255, 200, 100)

# Цвет текста (B, G, R)
COLOR_TEXT = (0, 0, 0)

# Толщина рамки
FRAME_WIDTH = 2

# Толщина текста
TEXT_WIDTH = 2


def draw(
    frame: np.ndarray,
    object_boxes: List[Tuple[int, int, int, int]],
    detected_indexes: List[int],
    detected_objects_scores: List[float],
    coco_classes: List[str],
) -> np.ndarray:
    """Рисует текст и рамки, вокруг обнаруженных объектов

    Args:
        frame (np.ndarray): текущий кадр
        object_boxes (List[Tuple[int, int, int, int]]): неотфильтрованный список локализацонных прямоугольников
        detected_indexes (List[int]): список индексов обнаруженных объектов
        detected_objects_scores (List[float]): список оценок (процентов) обнаруженных объектов
        coco_classes (List[str]): список доступных классов модели

    Returns:
        np.ndarray: кадр с аннотациями
    """
    # Отбор ограничивающих рамок с порогом nms_threshold > 0.4 (позволяет избежать повторяющихся рамок)
    object_boxes_filtered = cv2.dnn.NMSBoxes(object_boxes, detected_objects_scores, 0.0, 0.4)

    for box_index in list(object_boxes_filtered):
        # Получение индекса класса
        class_index = detected_indexes[box_index]

        # Раскомментируйте код ниже, чтобы исключить зебр из аннотаций (конспирация)
        # TODO: УДАЛИТЬ ЭТОТ КОД ПЕРЕД ВСТАВКОЙ В ОТЧЁТ
        # if coco_classes[class_index] == "zebra":
        #    continue

        # Рамка и текст класса
        x, y, w, h = object_boxes[box_index]

        # Рамка
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_BACK, FRAME_WIDTH)

        # Текст
        frame_text = f"{coco_classes[class_index].upper()}: {detected_objects_scores[box_index] * 100.0:.0f}%"

        # Фон текста
        frame = cv2.rectangle(
            frame,
            (x - FRAME_WIDTH // 2, y - 20),
            (x - FRAME_WIDTH // 2 + len(frame_text) * 10, y),
            COLOR_BACK,
            -1,
        )

        # Текст
        frame = cv2.putText(frame, frame_text, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, COLOR_TEXT, TEXT_WIDTH)

    return frame
