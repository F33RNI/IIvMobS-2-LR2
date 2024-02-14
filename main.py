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
import numpy as np


def main() -> None:
    """Main program entry"""
    # Initialize logging
    logging.basicConfig(level=logging.INFO)

    # Load model class names
    file = open(os.path.join("model", "classes.txt"), "r", encoding="utf-8")
    classes = file.read().split("\n")
    file.close()
    logging.info("Loaded classes.txt")

    # Load YOLO weights and config
    model = cv2.dnn.readNetFromDarknet(
        os.path.join("model", "model.cfg"),
        os.path.join("model", "model.weights"),
    )
    logging.info("Loaded model.cfg, model.weights")

    # Parse model layers
    layer_names = model.getLayerNames()
    out_layers_indexes = model.getUnconnectedOutLayers()
    out_layers = [layer_names[index - 1] for index in out_layers_indexes]

    # Start OpenCV video cap
    cap = cv2.VideoCapture("photos/giraffe_and_persons.jpg")

    # Main loop
    while True:
        # Capture the video image by image
        ret, image = cap.read()

        # Exit if error or image is None
        if not ret or image is None:
            # Wait for any key and exit
            logging.info("Press eny key to exit")
            cv2.waitKey(0)
            break

        # Retrieve image's dimentions
        frame_h, frame_w, _ = image.shape

        # Resize image to 608x608 and convert BGR to RGB and divide each pixel value to 255
        image_blob = cv2.dnn.blobFromImage(image, 1 / 255, (608, 608), (0, 0, 0), swapRB=True, crop=False)

        # Feed image to model
        model.setInput(image_blob)
        model_outputs = model.forward(out_layers)

        # Parse results into class_indexes, class_scores and bounding object_boxes
        class_indexes, class_scores, object_boxes = ([] for _ in range(3))
        for model_output in model_outputs:
            for detected_object in model_output:
                scores = detected_object[5:]
                object_class_index = np.argmax(scores)
                class_score = scores[object_class_index]
                if class_score > 0:
                    center_x = int(detected_object[0] * frame_w)
                    center_y = int(detected_object[1] * frame_h)
                    obj_width = int(detected_object[2] * frame_w)
                    obj_height = int(detected_object[3] * frame_h)
                    box = [center_x - obj_width // 2, center_y - obj_height // 2, obj_width, obj_height]
                    object_boxes.append(box)
                    class_indexes.append(object_class_index)
                    class_scores.append(float(class_score))

        # prevent duplicated detections
        filtered_boxes = cv2.dnn.NMSBoxes(object_boxes, class_scores, 0.0, 0.4)

        # List each detected and filtered object
        for box_index in list(filtered_boxes):
            # Extract class index
            object_class_index = class_indexes[box_index]

            # Uncomment code below to exclude giraffe from annotations
            # TODO: REMOVE THIS CODE
            # if classes[object_class_index] == "giraffe":
            #   continue

            # Draw bounding box
            x, y, w, h = object_boxes[box_index]
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

            # Draw text
            image = cv2.putText(
                image,
                f"{classes[object_class_index]}: {(class_scores[box_index] * 100):.2f}%",
                (x, y - 15),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 255, 255),
                2,
            )

        # open image as window
        cv2.imshow("image", image)

        # Press ESC to quit
        if cv2.waitKey(30) & 0xFF == 27:
            break

    # Release file / close camera
    cap.release()

    # Destroy OpenCV windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
