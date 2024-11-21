import pandas as pd
import cv2
import numpy as np
import torch
import warnings
import time
from flirpy.camera.lepton import Lepton
import threading
import copy
import serial

# 경고 메시지 숨기기
warnings.filterwarnings("ignore", category=FutureWarning)

# 부저를 Arduino와 시리얼 통신을 통해 제어
ser = serial.Serial('COM8', 9600)  # Arduino가 연결된 COM 포트 설정

# 엑셀 파일에서 데이터 읽기
df = pd.read_excel('dot_pattern.xlsx')
rgb_points = df[['RGB_X', 'RGB_Y']].values
thermal_points = df[['Thermal_X', 'Thermal_Y']].values

# 호모그래피 매트릭스 계산
H, _ = cv2.findHomography(rgb_points, thermal_points, method=cv2.RANSAC)

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)

# 쓰러짐 감지 및 알람용 변수 초기화
fall_detected_time = None
ALARM_DURATION = 5  # 5초 지속 시 알람

# 화재 감지 온도 임계값
FIRE_THRESHOLD = 70

# 두 카메라 캡처 (RGB 카메라)
cap_rgb = cv2.VideoCapture(2)

# 안전 구역 설정 변수 초기화
safety_zone = []
person_centers = {}
alert_times = {}

# 부저 알람 트리거 함수
def trigger_buzzer():
    for _ in range(2):  # 부저 알람을 2회 울림
        ser.write(b'1')  # 부저 ON 신호 전송
        time.sleep(0.1)  # 부저 울림 지속 시간
        ser.write(b'0')  # 부저 OFF 신호 전송
        time.sleep(0.1)  # 부저 울림 간격

# 마우스 콜백 함수: 안전 구역을 다각형으로 정의하거나 초기화하는 역할
def mouse_callback(event, x, y, flags, param):
    global safety_zone
    if event == cv2.EVENT_LBUTTONDOWN:
        safety_zone.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        safety_zone = []  # 우클릭 시 안전 구역 초기화

cv2.namedWindow('YOLOv5 Object Detection', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('YOLOv5 Object Detection', mouse_callback)

# 열화상 카메라 클래스 정의
class FlirLeptonCameraController(object):
    def __init__(self, device_name):
        self.thread_lock = threading.Lock()
        self.t_image_data = None
        self.is_thread_running = False
        self.is_camera_ready = False
        self.device_name = device_name
        self.camera = None
        self.alarm_triggered = False

    def get_recent_data(self):
        with self.thread_lock:
            return self.t_image_data

    def write_current_data(self, data):
        with self.thread_lock:
            self.t_image_data = copy.deepcopy(data)

    def image_capture(self):
        self.is_camera_ready = True
        while self.is_thread_running and self.is_camera_ready:
            if self.camera is not None:
                image = self.camera.grab().astype(np.float32)
                if image is not None:
                    self.write_current_data(image)
                time.sleep(0.03)
            else:
                break

    def start(self):
        self.camera = Lepton()
        if self.camera is not None:
            self.is_thread_running = True
            self.capture_thread = threading.Thread(target=self.image_capture)
            self.capture_thread.daemon = False
            self.capture_thread.start()
        else:
            print("Thermal camera is not connected.")

    def stop(self):
        if self.capture_thread is not None:
            self.is_thread_running = False
            self.capture_thread.join()
        if self.camera is not None:
            self.camera.close()
            self.camera = None

# 열화상 카메라 초기화 및 시작
thermal_camera = FlirLeptonCameraController('lepton_thermal_image')
thermal_camera.start()

while True:
    # RGB 이미지 캡처
    ret_rgb, frame_rgb = cap_rgb.read()
    if not ret_rgb:
        print("Unable to retrieve frame from RGB camera.")
        break

    # Thermal 이미지 캡처
    frame_thermal = thermal_camera.get_recent_data()
    if frame_thermal is None:
        print("Thermal camera not ready.")
        time.sleep(1)
        continue

    # 밝기 확인
    avg_brightness = np.mean(frame_rgb) if ret_rgb else 0

    # 낮과 밤 판단
    if avg_brightness < 50:  # 어두운 환경
        print("Low light detected: Using thermal camera.")
        # 열화상 데이터를 YOLO에 입력
        frame_thermal_normalized = cv2.normalize(frame_thermal, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        frame_thermal_colored = cv2.applyColorMap(frame_thermal_normalized, cv2.COLORMAP_JET)
        results = model(frame_thermal_colored)
        detection_frame = frame_thermal_colored
    else:  # 밝은 환경
        print("Sufficient light detected: Using RGB camera.")
        results = model(frame_rgb)
        detection_frame = frame_rgb

    # YOLO 탐지 결과 처리
    for i, result in enumerate(results.xyxy[0]):
        x1, y1, x2, y2, confidence, cls = map(int, result[:6])
        if cls == 0:  # 사람만 탐지
            color = (0, 255, 0)  # 초록색
            label = f"Person {confidence:.2f}"
            cv2.rectangle(detection_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(detection_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 안전 구역 표시
    if len(safety_zone) >= 3:
        pts = np.array(safety_zone, np.int32).reshape((-1, 1, 2))
        cv2.polylines(detection_frame, [pts], isClosed=True, color=(255, 0, 0), thickness=1)

    # 결과 이미지 출력
    cv2.imshow("YOLOv5 Object Detection", detection_frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap_rgb.release()
thermal_camera.stop()
cv2.destroyAllWindows()
ser.close()
