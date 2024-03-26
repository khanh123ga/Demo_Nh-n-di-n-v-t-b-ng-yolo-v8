from ultralytics import YOLO
import numpy

model = YOLO('D:\Yolo8\yolov8n.pt')  

# predict on an image
detection_output = model.predict(source="D:/Yolo8/imge/test/khanh/image_2.jpg", conf=0.25, save=False) 

# Display tensor array
print(detection_output)

# Display numpy array
print(detection_output[0].numpy())