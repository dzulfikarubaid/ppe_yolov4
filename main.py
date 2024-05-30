import cv2
import numpy as np

model = cv2.dnn.readNet('model.cfg', 'model.weights')
image = cv2.imread('image.png')
image = cv2.resize(image, (416, 416))
w, h, _ = image.shape
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
model.setInput(blob)
image_net_names = ["Ear muff", "Glasses", "Gloves", "Helm", "Masker", "Rompi", "safety shoes"]
class_names = [name.split(',')[0] for name in image_net_names]
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))
outputs = model.forward()
for i in np.arange(0, outputs.shape[2]):
   # extract the confidence of the i
   confidence = outputs[0, 0, i, 2]
   print(confidence)
   # draw bounding boxes only if the detection confidence is above...
   # ... a certain threshold, else skip
   if confidence > 0.3:
    idx = int(outputs[0, 0, i, 1])
    box = outputs[0, 0, i, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    # display the prediction
    label = "{}: {:.2f}%".format(class_names[idx], confidence * 100)
    print("[INFO] {}".format(label))
    cv2.rectangle(image, (startX, startY), (endX, endY),
        COLORS[idx], 2)
    y = startY - 15 if startY - 15 > 15 else startY + 15
    cv2.putText(image, label, (startX, y),
    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
  
cv2.imshow('image', image)
cv2.imwrite('image_result.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()