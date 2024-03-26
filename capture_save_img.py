import cv2

def open_save_camera(num_img,path):
    # Khởi tạo camera
    cap = cv2.VideoCapture(0)  

    if not cap.isOpened():
        print("Không thể mở camera.")
        return

    while True:
        # Đọc frame từ camera
        ret, frame = cap.read()

        if not ret:
            print("Không thể đọc frame từ camera.")
            break

        # Hiển thị frame
        cv2.imshow('Camera', frame)
        for i in range(num_img):
         image_path = f"{path}/image_{i+1}.jpg"
         cv2.imwrite(image_path,frame)
        # Nhấn 'o' để thoát
        if cv2.waitKey(1) & 0xFF == ord('o'):
            break
    

    # Giải phóng camera và đóng cửa sổ hiển thị
    cap.release()
    print('Ảnh đã được chụp và lưu')
    cv2.destroyAllWindows()

# Sử dụng hàm để mở camera
num = 10
save_path = 'D:\Yolo8\imge'
open_save_camera(num,save_path)
