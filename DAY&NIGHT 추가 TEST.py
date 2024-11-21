import pandas as pd
import cv2
import numpy as np
import torch
import datetime
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

cv2.namedWindow('YOLOv5 Object Detection - Person Only', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('YOLOv5 Object Detection - Person Only', mouse_callback)

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

# 밝기 수준 측정 함수
def get_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.mean()

while True:
    # RGB 이미지 캡처
    ret_rgb, frame_rgb = cap_rgb.read()
    if not ret_rgb:
        print("Unable to retrieve frame from RGB camera.")
        break

    # Thermal 이미지 캡처
    frame_thermal = thermal_camera.get_recent_data()
    if frame_thermal is None:
        print("Start FireDetection System")
        time.sleep(1)
        continue

    # 밝기 수준 계산
    brightness = get_brightness(frame_rgb)
    is_night = brightness < 50  # 밝기가 50 이하이면 '밤'으로 판단

    # 열화상 온도 변환
    img_celsius = frame_thermal / 100 - 273.15
    max_temp = img_celsius.max()

    # 화재 감지 및 부저 알람
    fire_mask = img_celsius >= FIRE_THRESHOLD
    if max_temp >= FIRE_THRESHOLD and not thermal_camera.alarm_triggered:
        print(f"Fire Detected! Current MaxTemp: {max_temp:.2f}°C")
        trigger_buzzer()
        thermal_camera.alarm_triggered = True
    elif max_temp < FIRE_THRESHOLD:
        thermal_camera.alarm_triggered = False

    # 밤과 낮에 따라 입력 이미지 결정
    if is_night:
        # 밤에는 열화상을 색상 모드로 변환
        frame_thermal_colored = cv2.applyColorMap(frame_thermal.astype(np.uint8), cv2.COLORMAP_HOT)
        input_image = frame_thermal_colored
    else:
        # 낮에는 RGB+열화상 정합
        aligned_rgb = cv2.warpPerspective(frame_rgb, H, (frame_thermal.shape[1], frame_thermal.shape[0]))
        gray_rgb = cv2.cvtColor(aligned_rgb, cv2.COLOR_BGR2GRAY)
        gray_rgb_colored = cv2.cvtColor(gray_rgb, cv2.COLOR_GRAY2BGR)

        # 화재 영역 강조
        red_outline = gray_rgb_colored.copy()
        contours, _ = cv2.findContours(fire_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(red_outline, contours, -1, (0, 0, 255), 2)

        # 화재 강조 결과 합성
        input_image = cv2.addWeighted(gray_rgb_colored, 1.0, red_outline, 0.7, 0)

    # YOLOv5 모델로 객체 탐지
    results = model(input_image)

    # 탐지된 사람 위치 초기화 및 카운트
    new_centers = {}
    person_count = 0
    fall_detected = False

    # 사람만 필터링하여 탐지 결과 표시
    for i, result in enumerate(results.xyxy[0]):
        x1, y1, x2, y2, confidence, cls = map(int, result[:6])

        if cls == 0:  # 사람만 감지
            person_count += 1
            width = x2 - x1
            height = y2 - y1
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            aspect_ratio = width / height
            person_id = f"person_{i + 1}"
            new_centers[person_id] = (center_x, center_y, width, height)

            # 바운딩 박스와 레이블 표시
            if aspect_ratio > 1.5:  # 비율 기준으로 누워있는 사람 판단
                label = f"Lying Person {confidence:.2f}"
                color = (0, 0, 255)  # 빨간색 (누워있는 사람)
                fall_detected = True  # 쓰러진 사람 감지 표시
            else:
                label = f"Person {confidence:.2f}"
                color = (0, 255, 0)  # 초록색 (서있는 사람)

            cv2.rectangle(input_image, (x1, y1), (x2, y2), color, 1)

    # 쓰러진 사람이 감지된 경우 알람 조건
    if fall_detected:
        if fall_detected_time is None:
            fall_detected_time = time.time()  # 쓰러진 상태 시작 시간 기록
        elif time.time() - fall_detected_time >= ALARM_DURATION:
            trigger_buzzer()
            print("Fall Detection")
    else:
        fall_detected_time = None  # 쓰러진 사람이 감지되지 않으면 시간 초기화

    # 사람 수를 화면에 표시
    cv2.putText(input_image, f"Count: {person_count}", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    person_centers = new_centers

    # 안전 구역 표시
    if len(safety_zone) >= 3:
        pts = np.array(safety_zone, np.int32).reshape((-1, 1, 2))
        cv2.polylines(input_image, [pts], isClosed=True, color=(255, 0, 0), thickness=1)

    # 결과 이미지 출력
    cv2.imshow("YOLOv5 Object Detection - Person Only", input_image)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap_rgb.release()
thermal_camera.stop()
cv2.destroyAllWindows()
ser.close()
