import cv2
import time

CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 0, 255), (255, 255, 255)]

class_names = ["Ear muff", "Glasses", "Gloves", "Helm", "Masker", "Rompi", "safety shoes"]

vc = cv2.VideoCapture("safety.mp4")


net = cv2.dnn.readNet("model.weights", "model.cfg")
# Apabila menggunakan cuda
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

while cv2.waitKey(1) < 1:
    (_, frame) = vc.read()
    if not _:
        exit()

    start = time.time()
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    end = time.time()

    start_drawing = time.time()
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid)]
        label = "%s : %f" % (class_names[int(classid)], score)
        cv2.rectangle(frame, box, color, 1)
        cv2.putText(frame, label, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    end_drawing = time.time()
    
    fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (1 / (end - start), (end_drawing - start_drawing) * 1000)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.imshow("detections", frame)