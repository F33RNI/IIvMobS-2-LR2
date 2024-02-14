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

import cv2

from detection import apply_yolo_object_detection


def main():
    # Изображение / видеофайл / веб-стрим / изображение или индекс вебки (TO_DETECT = 0)
    video_capture = cv2.VideoCapture("photos/persons_cars_and_bear.jpg")
    while True:
        # Чтение кадра
        ret, frame = video_capture.read()

        # Выход при ошибке или если больше нет кадров
        if not ret or frame is None:
            cv2.waitKey(0)
            break

        # Распознавание и локализация
        apply_yolo_object_detection(frame)

        # Отрисовка
        cv2.imshow("LR2 jundevchik", frame)

        # Ожидание q для выхода
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
