from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import mediapipe as mp
from ultralytics import YOLO
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

app = Flask(__name__)

# --- KHỞI TẠO AI (Dùng đường dẫn tương đối để lên Cloud không bị lỗi) ---
print("Đang nạp bộ não AI...")
yolo_model = YOLO('best.pt') 
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options, num_hands=1,
    min_hand_detection_confidence=0.5, min_tracking_confidence=0.5)
detector = vision.HandLandmarker.create_from_options(options)

# Biến toàn cục để lưu trạng thái
output_text = ""
current_char = ""
last_prediction = ""
last_added_char = ""
consecutive_frames = 0


def get_hand_crop(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = detector.detect(mp_image)
    if not result.hand_landmarks: return None, None
    h, w, _ = frame.shape
    xs = [int(lm.x * w) for lm in result.hand_landmarks[0]]
    ys = [int(lm.y * h) for lm in result.hand_landmarks[0]]
    x_min, x_max = max(0, min(xs) - 20), min(w, max(xs) + 20)
    y_min, y_max = max(0, min(ys) - 20), min(h, max(ys) + 20)
    return frame[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)


@app.route('/')
def index():
    return render_template('index.html')


# API CHÍNH: Xử lý từng khung hình gửi từ Web
@app.route('/process_frame', methods=['POST'])
def process_frame():
    global output_text, current_char, last_prediction, last_added_char, consecutive_frames

    # 1. Nhận ảnh Base64 từ trình duyệt
    data = request.json['image']
    encoded_data = data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. Xử lý Logic AI (Giữ nguyên y hệt code xịn của bạn)
    frame = cv2.flip(frame, 1)  # Lật ảnh như soi gương
    crop_img, bbox = get_hand_crop(frame)
    temp_char = ""

    if crop_img is not None:
        results = yolo_model.predict(crop_img, conf=0.45, verbose=False)
        if len(results[0].boxes) > 0:
            class_id = int(results[0].boxes.cls[0])
            temp_char = yolo_model.names[class_id]
            current_char = temp_char

            # Vẽ khung
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 136), 2)
            cv2.putText(frame, temp_char, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 136), 2)

        if temp_char and temp_char.lower() not in ['nothing', 'none', 'background']:
            if temp_char != last_added_char:
                if temp_char == last_prediction:
                    consecutive_frames += 1
                else:
                    consecutive_frames = 0
                    last_prediction = temp_char

                label = temp_char.lower()
                frames_to_lock = 4 if label in ['space', 'del', 'delete', 'backspace'] else 7

                if consecutive_frames >= frames_to_lock:
                    if label == 'space':
                        output_text += " "
                    elif label in ['del', 'delete', 'backspace']:
                        output_text = output_text[:-1]
                    else:
                        output_text += temp_char

                    last_added_char = temp_char
                    consecutive_frames = 0
            else:
                consecutive_frames = 0
        else:
            last_added_char = ""
            last_prediction = ""
            consecutive_frames = 0
    else:
        last_added_char = ""
        consecutive_frames = 0

    # 3. Mã hóa lại ảnh đã vẽ khung thành Base64
    # Ép nén chất lượng jpg để truyền qua mạng nhanh hơn
    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    encoded_img = base64.b64encode(buffer).decode('utf-8')

    # 4. Trả ảnh và cả dòng chữ Text về cho Website
    return jsonify({
        'result_image': 'data:image/jpeg;base64,' + encoded_img,
        'translated_text': output_text,
        'current_char': current_char
    })


@app.route('/clear_data', methods=['GET'])
def clear_data():
    global output_text
    output_text = ""
    return jsonify({"status": "success"})


if __name__ == '__main__':
    app.run(debug=True, port=5000)

#from flask import Flask, render_template, Response, jsonify
# import cv2
# import mediapipe as mp
# from ultralytics import YOLO
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
#
# app = Flask(__name__)
#
# # --- KHỞI TẠO AI ---
# yolo_model = YOLO(r'D:\pbl5\k-neural-api\A_DATN\best.pt')
# base_options = python.BaseOptions(model_asset_path=r'D:\pbl5\k-neural-api\A_DATN\hand_landmarker.task')
# options = vision.HandLandmarkerOptions(
#     base_options=base_options, num_hands=1,
#     min_hand_detection_confidence=0.5, min_tracking_confidence=0.5)
# detector = vision.HandLandmarker.create_from_options(options)
#
# # Biến toàn cục để lưu trữ văn bản và trạng thái
# output_text = ""
# current_char = ""
# last_prediction = ""
# last_added_char = ""  # Biến khóa: Ngăn chặn lặp chữ AAAAA
# consecutive_frames = 0
#
#
# def get_hand_crop(frame):
#     img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
#     result = detector.detect(mp_image)
#     if not result.hand_landmarks: return None, None
#     h, w, _ = frame.shape
#     xs = [int(lm.x * w) for lm in result.hand_landmarks[0]]
#     ys = [int(lm.y * h) for lm in result.hand_landmarks[0]]
#     x_min, x_max = max(0, min(xs) - 20), min(w, max(xs) + 20)
#     y_min, y_max = max(0, min(ys) - 20), min(h, max(ys) + 20)
#     return frame[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)
#
#
# def gen_frames():
#     global output_text, current_char, last_prediction, last_added_char, consecutive_frames
#     cap = cv2.VideoCapture(0)
#
#     # Ép camera chạy ở độ phân giải thấp hơn một chút để AI xử lý mượt mà hơn (giảm độ trễ)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#
#     while True:
#         success, frame = cap.read()
#         if not success: break
#
#         frame = cv2.flip(frame, 1)
#         crop_img, bbox = get_hand_crop(frame)
#         temp_char = ""
#
#         if crop_img is not None:
#             # THÊM conf=0.45: Yêu cầu AI phải chắc chắn >45% mới nhận diện, loại bỏ các frame nhiễu
#             results = yolo_model.predict(crop_img, conf=0.45, verbose=False)
#             if len(results[0].boxes) > 0:
#                 class_id = int(results[0].boxes.cls[0])
#                 temp_char = yolo_model.names[class_id]
#                 current_char = temp_char
#
#                 # Vẽ box
#                 cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 136), 2)
#                 cv2.putText(frame, temp_char, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 136), 2)
#
#             if temp_char and temp_char.lower() not in ['nothing', 'none', 'background']:
#
#                 if temp_char != last_added_char:
#                     if temp_char == last_prediction:
#                         consecutive_frames += 1
#                     else:
#                         consecutive_frames = 0
#                         last_prediction = temp_char
#
#                     # --- LOGIC ĐỘ NHẠY BẤT ĐỐI XỨNG ---
#                     label = temp_char.lower()
#
#                     # Quy định số frame cần thiết để chốt chữ (càng nhỏ càng nhạy)
#                     if label in ['space', 'del', 'delete', 'backspace']:
#                         frames_to_lock = 4  # Space và Del cực kỳ nhạy, xẹt qua xíu là ăn ngay
#                     else:
#                         frames_to_lock = 7  # Chữ cái bình thường giảm từ 15 xuống 7 (nhanh gấp đôi)
#
#                     # Đạt ngưỡng ổn định
#                     if consecutive_frames >= frames_to_lock:
#                         if label == 'space':
#                             output_text += " "
#                         elif label in ['del', 'delete', 'backspace']:
#                             output_text = output_text[:-1]
#                         else:
#                             output_text += temp_char
#
#                         last_added_char = temp_char
#                         consecutive_frames = 0
#                 else:
#                     consecutive_frames = 0
#             else:
#                 last_added_char = ""
#                 last_prediction = ""
#                 consecutive_frames = 0
#         else:
#             last_added_char = ""
#             consecutive_frames = 0
#
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame_bytes = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#
# # --- ĐỊNH TUYẾN (ROUTES) ---
# @app.route('/')
# def index():
#     return render_template('index.html')
#
#
# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
#
#
# @app.route('/get_data')
# def get_data():
#     return jsonify({
#         "translated_text": output_text,
#         "current_char": current_char
#     })
#
#
# @app.route('/clear_data')
# def clear_data():
#     global output_text
#     output_text = ""
#     return jsonify({"status": "success"})
#
#
# if __name__ == '__main__':
#     app.run(debug=True, port=5000)
#
#---------------------------------------
#
# # # from flask import Flask, render_template, Response, request, jsonify
# # # import cv2
# # # import os
# # # from ultralytics import YOLO
# # #
# # # app = Flask(__name__)
# # # UPLOAD_FOLDER = 'uploads'
# # # os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# # #
# # # # Load model đã train với mAP 0.995
# # # model = YOLO(r"D:\pbl5\k-neural-api\A_DATN\_output_\runs\gesture_train_v1\weights\best.pt")
# # #
# # # # Biến lưu trữ trạng thái
# # # current_source = 0
# # # spelling_text = ""
# # # last_char = ""
# # # char_count = 0
# # # hand_sign_classes = list(range(29))
# # #
# # #
# # # def generate_frames():
# # #     global current_source, spelling_text, last_char, char_count
# # #     cap = cv2.VideoCapture(current_source)
# # #
# # #     while True:
# # #         success, frame = cap.read()
# # #         if not success:
# # #             if isinstance(current_source, str):
# # #                 cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
# # #                 continue
# # #             break
# # #
# # #         # --- BƯỚC 1: ĐỒNG NHẤT KÍCH THƯỚC (QUAN TRỌNG) ---
# # #         # Resize mọi nguồn về 640x480 để AI nhận diện ổn định như nhau
# # #         frame = cv2.resize(frame, (640, 480))
# # #         annotated_frame = frame.copy()
# # #
# # #         # --- BƯỚC 2: TỐI ƯU ĐỘ NHẠY ---
# # #         # Giảm conf xuống 0.4 để nhạy hơn với video tải lên
# # #         # Giữ imgsz=224 theo đúng Notebook bạn đã train
# # #         results = model(frame, stream=True, imgsz=224, conf=0.4, classes=hand_sign_classes)
# # #
# # #         # Vẽ thanh đen hiển thị Spelling lên màn hình
# # #         cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 60), (0, 0, 0), -1)
# # #         cv2.putText(annotated_frame, f"ASL Spelling: {spelling_text}", (15, 45),
# # #                     cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
# # #
# # #         found_hand = False
# # #         for r in results:
# # #             if len(r.boxes) > 0:
# # #                 found_hand = True
# # #                 annotated_frame = r.plot()
# # #
# # #                 cls_id = int(r.boxes.cls[0])
# # #                 label = model.names[cls_id]
# # #
# # #                 if label != "nothing":
# # #                     if label == last_char:
# # #                         char_count += 1
# # #                     else:
# # #                         last_char = label
# # #                         char_count = 0
# # #
# # #                     # Tăng lên 15 để chữ không bị nhảy quá nhanh khi nhạy hơn
# # #                     if char_count == 15:
# # #                         if label == "space":
# # #                             spelling_text += " "
# # #                         elif label == "del":
# # #                             spelling_text = spelling_text[:-1]
# # #                         else:
# # #                             spelling_text += label
# # #                         char_count = 0
# # #
# # #         if not found_hand:
# # #             last_char = ""
# # #
# # #         ret, buffer = cv2.imencode('.jpg', annotated_frame)
# # #         yield (b'--frame\r\n'
# # #                b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
# # #     cap.release()
# # #
# # #
# # # @app.route('/upload_video', methods=['POST'])
# # # def upload_video():
# # #     global current_source, spelling_text
# # #     if 'video' not in request.files: return jsonify(success=False)
# # #     file = request.files['video']
# # #     if file.filename == '': return jsonify(success=False)
# # #
# # #     video_path = os.path.join(UPLOAD_FOLDER, file.filename)
# # #     file.save(video_path)
# # #     current_source = video_path
# # #     spelling_text = ""
# # #     return jsonify(success=True)
# # #
# # #
# # # @app.route('/use_camera', methods=['POST'])
# # # def use_camera():
# # #     global current_source, spelling_text
# # #     current_source = 0
# # #     spelling_text = ""
# # #     return jsonify(success=True)
# # #
# # #
# # # @app.route('/video_feed')
# # # def video_feed():
# # #     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
# # #
# # #
# # # @app.route('/')
# # # def index():
# # #     return render_template('index.html')
# # #
# # #
# # # if __name__ == '__main__':
# # #     app.run(debug=True, use_reloader=False)
# #
# # from flask import Flask, render_template, Response
# # import cv2
# # from ultralytics import YOLO
# #
# # app = Flask(__name__)
# #
# # print("🚀 ĐANG TẢI MÔ HÌNH YOLO CÓ VÙNG QUÉT ROI...")
# # # Cập nhật đường dẫn file weights của bạn tại đây nếu cần
# # model = YOLO(r'D:\pbl5\k-neural-api\A_DATN\best.pt')
# # class_names = model.names
# #
# #
# # def generate_frames():
# #     # Sử dụng camera mặc định. Nếu dùng camera ngoài, hãy thử đổi số 0 thành 1.
# #     cap = cv2.VideoCapture(0)
# #
# #     while True:
# #         success, frame = cap.read()
# #         if not success:
# #             break
# #
# #         # Lật ảnh theo chiều ngang (giống soi gương)
# #         frame = cv2.flip(frame, 1)
# #         h, w, _ = frame.shape
# #         annotated_frame = frame.copy()
# #
# #         # ---------------------------------------------------------
# #         # 1. XÁC ĐỊNH VÙNG QUÉT CỐ ĐỊNH (ROI - Region Of Interest)
# #         # Kích thước khung quét là 300x300 pixel, nằm giữa màn hình
# #         # ---------------------------------------------------------
# #         box_size = 300
# #         x1 = int(w / 2 - box_size / 2)
# #         y1 = int(h / 2 - box_size / 2)
# #         x2 = x1 + box_size
# #         y2 = y1 + box_size
# #
# #         # Cắt bức ảnh lớn thành một bức ảnh nhỏ (chỉ chứa vùng bên trong khung)
# #         roi_frame = frame[y1:y2, x1:x2]
# #
# #         # --- BẮT ĐẦU PHẦN THAY ĐỔI ---
# #         # Hai dòng dưới đây đã được chú thích để KHÔNG vẽ khung xanh dương nữa.
# #         # cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
# #         # cv2.putText(annotated_frame, "DAT TAY VAO KHUNG NAY", (x1, y1 - 10),
# #         #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
# #         # --- KẾT THÚC PHẦN THAY ĐỔI ---
# #
# #         # ---------------------------------------------------------
# #         # 2. CHỈ ĐƯA VÙNG ROI CHO YOLO DỰ ĐOÁN
# #         # ---------------------------------------------------------
# #         # conf=0.6: Độ tự tin phải trên 60% mới lấy
# #         results = model.predict(source=roi_frame, conf=0.6, verbose=False)
# #
# #         # Nếu YOLO tìm thấy bàn tay TRONG vùng ROI
# #         if len(results[0].boxes) > 0:
# #             # Lấy kết quả tự tin nhất
# #             box = results[0].boxes[0]
# #             conf = float(box.conf[0])
# #             cls_id = int(box.cls[0])
# #             label_name = class_names[cls_id]
# #
# #             # Lấy tọa độ bàn tay mà YOLO tìm thấy (Lưu ý: Tọa độ này là so với cái khung nhỏ)
# #             bx1, by1, bx2, by2 = map(int, box.xyxy[0])
# #
# #             # Quy đổi tọa độ nhỏ về tọa độ của màn hình lớn để vẽ
# #             real_x1 = x1 + bx1
# #             real_y1 = y1 + by1
# #             real_x2 = x1 + bx2
# #             real_y2 = y1 + by2
# #
# #             # Đổi màu khung nhận diện thành Xanh Lá, báo hiệu thành công
# #             cv2.rectangle(annotated_frame, (real_x1, real_y1), (real_x2, real_y2), (0, 255, 0), 3)
# #
# #             # Hiển thị chữ cái ASL
# #             text = f"ASL: {label_name} ({conf * 100:.0f}%)"
# #             (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
# #             cv2.rectangle(annotated_frame, (real_x1, real_y1 - 30), (real_x1 + text_w, real_y1), (0, 0, 0), -1)
# #             cv2.putText(annotated_frame, text, (real_x1, real_y1 - 5),
# #                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
# #
# #         # Xuất video ra web dưới dạng stream MJPEG
# #         ret, buffer = cv2.imencode('.jpg', annotated_frame)
# #         frame_bytes = buffer.tobytes()
# #
# #         yield (b'--frame\r\n'
# #                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
# #
# #     cap.release()
# #
# #
# # @app.route('/')
# # def index():
# #     return render_template('index.html')
# #
# #
# # @app.route('/video_feed')
# # def video_feed():
# #     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
# #
# #
# # if __name__ == "__main__":
# #     # Chạy server Flask trên port 5000. debug=True giúp tự load lại khi sửa code.
# #     app.run(host='0.0.0.0', port=5000, debug=True)
