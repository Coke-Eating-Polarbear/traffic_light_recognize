import cv2
import numpy as np
import time

start = time.time()

net = cv2.dnn.readNet("yolov3_custom_final.weights", "yolov3_custom.cfg") #가중치 가져오기

classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

image = cv2.imread("09876.png") #사진 입력

blob = cv2.dnn.blobFromImage(image, 1 / 255, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
outs = net.forward(net.getUnconnectedOutLayersNames())

conf_threshold = 0.5  # Confidence(신뢰도) 임계값
nms_threshold = 0.4  # Non-maximum suppression(비최대 억제) 임계값

lower_red = np.array([0, 30, 30])  # 빨간색 범위의 하한값
upper_red = np.array([20, 255, 255])  # 빨간색 범위의 상한값
lower_orange = np.array([21, 30, 30])  # 주황색 범위의 하한값
upper_orange = np.array([45, 255, 255])  # 주황색 범위의 상한값
lower_green = np.array([15, 30, 30])  # 초록색 범위의 하한값
upper_green = np.array([105, 255, 255])  # 초록색 범위의 상한값

class_ids = []
confidences = []
boxes = []
colors = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > conf_threshold:
            center_x = int(detection[0] * image.shape[1])
            center_y = int(detection[1] * image.shape[0])
            width = int(detection[2] * image.shape[1])
            height = int(detection[3] * image.shape[0])

            x = int(center_x - width / 2)
            y = int(center_y - height / 2)

            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, width, height])

# 비최대 억제 적용
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

if len(indices) > 0:
    for i in indices:
        if i < len(class_ids):
            box = boxes[i]
            x, y, width, height = box[0], box[1], box[2], box[3]
            total_pixels = width * height
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"

            object_roi = image[y:y + height, x:x + width]
            hsv_roi = cv2.cvtColor(object_roi, cv2.COLOR_BGR2HSV)

            mask_red = cv2.inRange(hsv_roi, lower_red, upper_red)
            mask_orange = cv2.inRange(hsv_roi, lower_orange, upper_orange)
            mask_green = cv2.inRange(hsv_roi, lower_green, upper_green)

            red_pixels = cv2.countNonZero(mask_red)
            orange_pixels = cv2.countNonZero(mask_orange)
            green_pixels = cv2.countNonZero(mask_green)

            red_count = red_pixels / total_pixels
            orange_count = orange_pixels / total_pixels
            green_count = green_pixels / total_pixels

            colors.append([red_count, orange_count, green_count])

            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

            if len(colors) > 0:
                labels = []
                sound = []

                print(red_count)
                print(orange_count)
                print(green_count)

                if 0.05<=red_count:
                    sound.append(330)
                    if red_count <= 0.2:
                        labels.append("Red Arrow")
                    if red_count >= 0.25:
                        labels.append("Red")
                if 0.05<=orange_count:
                    sound.append(294)
                    if orange_count <= 0.2:
                        labels.append("Orange Arrow")
                    if orange_count >= 0.25:
                        labels.append("Orange")
                if 0.05<=green_count:
                    sound.append(262)
                    if green_count <= 0.2:
                        labels.append("Green Arrow")
                    if green_count >= 0.25:
                        labels.append("Green")

                if not labels:
                    sound.append(0)
                    labels.append("Off")

                label = ", ".join(labels)
            cv2.putText(image, f"Colors: {label}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

end = time.time()
print(f"{end - start:.5f} sec")
cv2.imshow("Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()



