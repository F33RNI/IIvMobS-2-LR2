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
import sys
from typing import List, Tuple

import cv2
import numpy as np

# Path to input image, video file or integer (0..) to use webcam instead
PHOTO_VIDEO_INPUT = "videos/planes_stock.mp4"

# Path to yolo directory with yolov4-tiny.cfg and yolov4-tiny.cfg files
YOLO_DIR = "yolov4-tiny"

# Files with YOLO class names inside YOLO_DIR
YOLO_CLASSES_FILE = "coco.names.txt"

# YOLO config file inside YOLO_DIR
YOLO_CFG_FILE = "yolov4-tiny.cfg"

# YOLO model weights file inside YOLO_DIR
YOLO_WEIGHTS_FILE = "yolov4-tiny.weights"


def logging_setup() -> None:
    """Sets up logging format and level"""

    # Create logs formatter
    log_formatter = logging.Formatter(
        "%(asctime)s %(threadName)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Setup logging into console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)

    # Add all handlers and setup level
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)

    # Log test message
    logging.info("Logging setup is complete")


def yolo_detect_and_localize(
    image_to_process: np.ndarray,
    net: cv2.dnn.Net,
    out_layers: List[str],
    classes: List[str],
    colors: List[Tuple[int, int, int]],
) -> np.ndarray:
    """Recognizes and localizes object on image

    Args:
        image_to_process (np.ndarray): source image
        net (cv2.dnn.Net): YOLO net object
        out_layers (List[str]): YOLO output layers list
        classes (List[str]): list of YOLO classes
        colors (List[Tuple(int, int, int)]): list of colors of each class

    Returns:
        np.ndarray: image with annotations
    """

    height, width, _ = image_to_process.shape

    # Prepare image
    blob = cv2.dnn.blobFromImage(image_to_process, 1 / 255, (608, 608), (0, 0, 0), swapRB=True, crop=False)

    # Feed image
    net.setInput(blob)
    outs = net.forward(out_layers)

    # Parse results into class_indexes, class_scores and bounding boxes
    class_indexes, class_scores, boxes = ([] for _ in range(3))
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

    # Keep boxes nms_threshold > 0.4
    nms_boxes = cv2.dnn.NMSBoxes(boxes, class_scores, 0.0, 0.4)

    # Draw annotation box and text for each class
    for box_index in list(nms_boxes):
        # Extract class index
        class_index = class_indexes[box_index]

        # Uncomment code below to exclude aeroplane from annotations
        # if classes[class_index] == "aeroplane":
        #    continue

        # Extract bounding box
        x, y, w, h = boxes[box_index]

        # Draw ROI
        image_to_process = cv2.rectangle(image_to_process, (x, y), (x + w, y + h), colors[class_index], 1)

        # Text
        roi_text = f"{classes[class_index]}: {(class_scores[box_index] * 100):.0f}%"

        # Draw text background
        image_to_process = cv2.rectangle(
            image_to_process,
            (x, y - 20),
            (x + len(roi_text) * 10, y),
            colors[class_index],
            -1,
        )

        # Draw annotation text
        image_to_process = cv2.putText(
            image_to_process,
            roi_text,
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255 - colors[class_index][0], 255 - colors[class_index][1], 255 - colors[class_index][2]),
            1,
            cv2.LINE_AA,
        )

    return image_to_process


def main() -> None:
    """Main program entry"""
    # Initialize logging
    logging_setup()

    # Load YOLO class names
    logging.info(f"Loading class names from {YOLO_CLASSES_FILE}")
    with open(os.path.join(YOLO_DIR, YOLO_CLASSES_FILE), "r", encoding="utf-8") as file:
        classes = file.read().split("\n")
        classes = [class_name.replace("\r", "") for class_name in classes if len(class_name.replace("\r", "")) > 1]

    # Create random color for each class
    colors = []
    for _ in range(len(classes)):
        color = np.random.randint(0, 256, size=(3,))
        colors.append((int(color[0]), int(color[1]), int(color[2])))

    # Load YOLO weights and config
    logging.info(f"Loading YOLO config from {YOLO_CFG_FILE} and weights from {YOLO_WEIGHTS_FILE}")
    net = cv2.dnn.readNetFromDarknet(
        os.path.join(YOLO_DIR, YOLO_CFG_FILE),
        os.path.join(YOLO_DIR, YOLO_WEIGHTS_FILE),
    )
    layer_names = net.getLayerNames()
    out_layers_indexes = net.getUnconnectedOutLayers()
    out_layers = [layer_names[index - 1] for index in out_layers_indexes]

    # Start OpenCV video capture
    capture = cv2.VideoCapture(PHOTO_VIDEO_INPUT)

    # Main loop
    while True:
        # Capture the video frame by frame
        _, frame = capture.read()

        # Exit if no frame
        if frame is None:
            # Wait for any key and exit
            logging.info("Done! Press any key to exit")
            cv2.waitKey(0)
            break

        # Detect, localize and annotate
        frame = yolo_detect_and_localize(frame, net, out_layers, classes, colors)

        # Display the resulting frame
        cv2.imshow("frame", frame)

        # 30 - 1/30 FPS
        # Press q to quit
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    # Release the cap object after the loop
    if capture is not None:
        capture.release()

    # Destroy all OpenCV windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
