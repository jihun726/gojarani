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

# 부저를 Arduino와 시리얼 통신으로 제어
ser = serial.Serial('COM8', 9600)  # Arduino가 연결된 포트 설정

# 엑셀 파일에서 대응점 데이터 읽기
df = pd.read_excel('dot_pattern.xlsx')
rgb_points = df[['RGB_X', 'RGB_Y']].values
thermal_points = df[['Thermal_X', 'Thermal_Y']].values

# 호모그래피 매트릭스 계산
H, _ = cv2.findHomography(rgb_points, thermal_points, method=cv2.RANSAC)

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)

# 설정값
FIRE_THRESHOLD = 40  # 화재 감지 온도 기준
FALL_RATIO_THRESHOLD = 1.5  # 쓰러짐 인식 비율 기준
ALARM_DURATION = 5  # 부저 알람 지속 시간
FALL_DETECTION_DURATION = 2  # 쓰러짐 지속 시간(초)
fall_detected_time = None

# RGB 카메라 초기화
cap_rgb = cv2.VideoCapture(2)

# 안전 구역 설정 변수
safety_zone = []
person_centers = {}
alert_times = {}

# 열화상 카메라 클래스
class FlirLeptonCameraController:
    def __init__(self):
        self.thread_lock = threading.Lock()
        self.t_image_data = None
        self.is_thread_running = False
        self.camera = None

    def get_recent_data(self):
        with self.thread_lock:
            return self.t_image_data

    def write_current_data(self, data):
        with self.thread_lock:
            self.t_image_data = copy.deepcopy(data)

    def image_capture(self):
        self.camera = Lepton()
        while self.is_thread_running:
            image = self.camera.grab().astype(np.float32)
            if image is not None:
                self.write_current_data(image)
            time.sleep(0.03)

    def start(self):
        self.is_thread_running = True
        self.capture_thread = threading.Thread(target=self.image_capture)
        self.capture_thread.start()

    def stop(self):
        self.is_thread_running = False
        self.capture_thread.join()
        if self.camera:
            self.camera.close()


# 열화상 카메라 초기화
thermal_camera = FlirLeptonCameraController()
thermal_camera.start()

# 부저 알람 트리거 함수
def trigger_buzzer():
    ser.write(b'1')  # 부저 ON
    time.sleep(0.1)
    ser.write(b'0')  # 부저 OFF
    time.sleep(0.1)

# 마우스 콜백 함수 (안전 구역 설정)
def mouse_callback(event, x, y, flags, param):
    global safety_zone
    if event == cv2.EVENT_LBUTTONDOWN:
        safety_zone.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        safety_zone = []  # 우클릭 시 안전 구역 초기화

cv2.namedWindow('YOLOv5 Object Detection - Person Only', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('YOLOv5 Object Detection - Person Only', mouse_callback)

# 메인 루프
while True:
    # RGB 이미지 캡처
    ret_rgb, frame_rgb = cap_rgb.read()
    if not ret_rgb:
        print("RGB 카메라에서 프레임을 읽을 수 없습니다.")
        break

    # 열화상 이미지 캡처
    frame_thermal = thermal_camera.get_recent_data()
    if frame_thermal is None:
        print("열화상 카메라 데이터 수신 중...")
        time.sleep(1)
        continue

    # 열화상 데이터를 섭씨 온도로 변환
    img_celsius = frame_thermal / 100 - 273.15
    max_temp = img_celsius.max()

    # 화재 감지 마스크 생성
    fire_mask = img_celsius >= FIRE_THRESHOLD

    # 호모그래피를 적용하여 RGB 이미지를 열화상 이미지 크기에 맞춤
    aligned_rgb = cv2.warpPerspective(frame_rgb, H, (frame_thermal.shape[1], frame_thermal.shape[0]))
    aligned_rgb = cv2.resize(aligned_rgb, (frame_thermal.shape[1], frame_thermal.shape[0]))

    # RGB 이미지를 그레이스케일로 변환 후 다시 BGR로 변환
    gray_rgb = cv2.cvtColor(aligned_rgb, cv2.COLOR_BGR2GRAY)
    gray_rgb_colored = cv2.cvtColor(gray_rgb, cv2.COLOR_GRAY2BGR)

    # 열화상 이미지를 uint8로 정규화
    frame_thermal_normalized = cv2.normalize(frame_thermal, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    thermal_colored = cv2.applyColorMap(frame_thermal_normalized, cv2.COLORMAP_JET)

    # 화재 영역 강조
    contours, _ = cv2.findContours(fire_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.drawContours(thermal_colored, [contour], -1, (0, 0, 255), 2)

    # 합성 이미지 생성
    result_image = cv2.addWeighted(gray_rgb_colored, 0.7, thermal_colored, 0.3, 0)

    # 화재 감지 알람
    if max_temp >= FIRE_THRESHOLD:
        print(f"화재 감지: 최대 온도 {max_temp:.2f}°C")
        trigger_buzzer()

    # YOLOv5로 사람 감지
    results = model(result_image)

    # 탐지된 객체 표시 및 쓰러짐 인식
    new_centers = {}
    person_count = 0
    fall_detected = False

    for detection in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = map(int, detection[:6])
        if cls == 0:  # 사람이 감지된 경우
            person_count += 1
            width = x2 - x1
            height = y2 - y1
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            aspect_ratio = width / height
            person_id = f"person_{person_count}"
            new_centers[person_id] = (center_x, center_y, width, height)

            # 쓰러짐 인식
            if aspect_ratio > FALL_RATIO_THRESHOLD:
                if fall_detected_time is None:
                    fall_detected_time = time.time()  # 쓰러짐 시작 시간 기록
                elif time.time() - fall_detected_time >= FALL_DETECTION_DURATION:
                    fall_detected = True
                    cv2.putText(result_image, "Fall Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                    trigger_buzzer()
            else:
                fall_detected_time = None

            # 바운딩 박스와 레이블 표시
            color = (0, 255, 0) if aspect_ratio <= FALL_RATIO_THRESHOLD else (0, 0, 255)
            label = f"Lying Person {conf:.2f}" if color == (0, 0, 255) else f"Person {conf:.2f}"
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(result_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 사람 수를 화면에 표시
    cv2.putText(result_image, f"Count: {person_count}", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # 안전 구역 표시
    if len(safety_zone) >= 3:
        pts = np.array(safety_zone, np.int32).reshape((-1, 1, 2))
        cv2.polylines(result_image, [pts], isClosed=True, color=(255, 0, 0), thickness=1)

    # 결과 이미지 출력
    cv2.imshow("YOLOv5 Object Detection - Person Only", result_image)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap_rgb.release()
thermal_camera.stop()
cv2.destroyAllWindows()
ser.close()
