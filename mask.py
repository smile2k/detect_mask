from argparse import ArgumentParser

import cv2

from utils import (
    CLASSES,
    COLORS,
    CONFIDENCE_THRESHOLD,
    NMS_THRESHOLD,
    draw_bounding_box_with_label,
    yolo_box_to_points,
)
'''
# parse the script parameters
parser = ArgumentParser(description="D:\\AI\\test\\anh_test15.jpg")
parser.add_argument(
    "--image", dest="D:\\AI\\test\\anh_test16.jpg", help="Path to the image", required=True
)
args = parser.parse_args()
image_path = args.image_path
'''
# load image
image = cv2.imread("D:\\AI\\test\\anh_test6.jpg")
height, width = image.shape[:2]

# load model
# (might need to change to absolute path rather than relative path)
weights_path = "D:\\AI\\yolov4\\yolov4-custom_best.weights"
config_path = "D:\\AI\\yolov4\\yolov4-custom.cfg"
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

# run model
model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
class_ids, scores, boxes = model.detect(
    image, confThreshold=CONFIDENCE_THRESHOLD, nmsThreshold=NMS_THRESHOLD
)

# draw bounding box for each object
for (class_id, score, box) in zip(class_ids, scores, boxes):
    label = "%s: %.2f" % (CLASSES[class_id], score)
    color = COLORS[class_id]
    x1, y1, x2, y2 = yolo_box_to_points(box)
    draw_bounding_box_with_label(image, x1, y1, x2, y2, label, color)

# display image
#image = cv2.resize(image, (0,0), fx=0.8, fy=0.8)
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()