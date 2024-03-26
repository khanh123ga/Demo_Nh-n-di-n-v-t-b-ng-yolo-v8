from ultralytics import YOLO 
import random
import cv2
import numpy as np 


#mở file txt
my_file = open("D:/Yolo8/text/test.txt", "r")
data = my_file.read()
class_list = data.split("\n")
my_file.close()

#tạo màu ngẫu nhiên
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

#model
model = YOLO('D:\Yolo8\yolov8n.pt','v8')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    #đọc frame camera của máy 
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    #  kích thước frame
    frame_width = 800
    frame_height = 600
    frame = cv2.resize(frame, (frame_width, frame_height))
    detect_params = model.predict(source=[frame], conf=0.45, save=False)

    # Chuyển  tensor thành array 
    DP = detect_params[0].numpy()
    print(DP)

    if len(DP) != 0:
        for i in range(len(DP)):
            print(i)
            boxes = detect_params[0].boxes
            box = boxes[i] 
            boxes = detect_params[0]
            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]



            x1 = max(0, int(bb[0]))  # Đảm bảo x1 không nhỏ hơn 0
            y1 = max(0, int(bb[1]))  # Đảm bảo y1 không nhỏ hơn 0
            x2 = min(frame_width - 1, int(bb[2]))  # Đảm bảo x2 không lớn hơn frame_width - 1
            y2 = min(frame_height - 1, int(bb[3]))  # Đảm bảo y2 không lớn hơn frame_height - 1
            cv2.rectangle(
               frame,
              (x1, y1),
              (x2, y2),
              detection_colors[int(clsID)],
              3,
            )

            # Name hiển thị phân loại
            font = cv2.FONT_HERSHEY_COMPLEX
            font_scale=1
            # Tính toán kích thước của văn bản
            text = class_list[int(clsID)] + " " + str(round(conf, 2)) + "%"
            text_size = cv2.getTextSize(text, font, font_scale, 1)[0]
    
            # Tính toán tọa độ của văn bản
            text_x = int(bb[0] + (bb[2] - bb[0]) / 2 - text_size[0] / 2)  # Giữa theo chiều ngang của hình chữ nhật
            text_y = int(bb[1]) - 5  # Trên hình chữ nhật
    
            # Đảm bảo văn bản không vượt ra ngoài khung hình
            text_x = max(0, min(text_x, frame_width - text_size[0] - 5))
            text_y = max(text_size[1], min(text_y, frame_height - 5))
            cv2.putText(
               frame,
               text,
               (text_x, text_y),
               font,
               font_scale,
               (255, 255, 255),
               2,
    )

    # show frame
   
    cv2.imshow("Nhận diện vật", frame)

    # nhấn 'o' để stop
    if cv2.waitKey(1) == ord("o"):
        break

# đóng cửa sổ 
cap.release()
cv2.destroyAllWindows()