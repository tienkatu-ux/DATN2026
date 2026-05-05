import eventlet
eventlet.monkey_patch() # Bắt buộc phải có để chạy SocketIO trên Render

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
from ultralytics import YOLO

app = Flask(__name__)
# Cấu hình SocketIO cho phép mọi kết nối
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# 1. NẠP BỘ NÃO AI 
yolo_model = YOLO('best.pt')
# (Nếu có MediaPipe, khai báo thêm ở đây)

@app.route('/')
def index():
    return render_template('index.html')

# 2. BỘ PHẬN LẮNG NGHE HÌNH ẢNH TỪ WEB GỬI LÊN
@socketio.on('image')
def handle_image(data):
    try:
        # Giải mã chuỗi base64 (hình ảnh) từ web thành định dạng ảnh của OpenCV
        encoded_data = data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # ========================================================
        # 3. CHỖ NÀY BẠN ĐƯA AI VÀO XỬ LÝ (Ví dụ của YOLO)
        results = yolo_model(frame, verbose=False)
        frame = results[0].plot() # YOLO tự vẽ khung nhận diện lên ảnh
        
        # (Nếu code cũ của bạn có MediaPipe vẽ tay, hãy ghép code vào đây)
        # ========================================================

        # Mã hóa lại bức ảnh đã vẽ kết quả thành base64 để gửi về web
        _, buffer = cv2.imencode('.jpg', frame)
        frame_encoded = base64.b64encode(buffer).decode('utf-8')
        image_data = "data:image/jpeg;base64," + frame_encoded
        
        # Bắn ngược ảnh về cho web hiển thị
        emit('response_back', image_data)

    except Exception as e:
        print(f"Lỗi xử lý khung hình: {e}")

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=10000)
