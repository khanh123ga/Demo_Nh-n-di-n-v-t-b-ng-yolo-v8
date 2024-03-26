
from ultralytics import YOLO
model = YOLO('weights/yolov8n-cls.pt') 
model.train(data='D:\Yolo8\imge', epochs=5,batch=5)  
model('D:\Yolo8\interface\img')
