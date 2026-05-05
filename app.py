from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
from ultralytics import YOLO

app = Flask(__name__)
# Đã đổi sang chế độ threading an toàn tuyệt đối cho AI
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# NẠP BỘ NÃO AI 
yolo_model = YOLO('best.pt')

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('image')
def handle_image(data):
    try:
        # Giải mã hình ảnh
        encoded_data = data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Đưa vào AI xử lý
        results = yolo_model(frame, verbose=False)
        frame = results[0].plot() 
        
        # Mã hóa và gửi về web
        _, buffer = cv2.imencode('.jpg', frame)
        frame_encoded = base64.b64encode(buffer).decode('utf-8')
        image_data = "data:image/jpeg;base64," + frame_encoded
        
        emit('response_back', image_data)

    except Exception as e:
        print(f"Lỗi xử lý khung hình: {e}")

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=10000)
