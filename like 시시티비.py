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

# 각 사람의 개별 영상을 표시할 윈도우 크기 설정
WINDOW_SIZE = (200, 200)  # 각 개별 영상의 크기 (200x200 픽셀)

# 부저 알람 트리거 함수
def trigger_buzzer():
    for _ in range(2):  # 부저 알람을 2회 울림
        ser.write(b'1')  # 부저 ON 신호 전송
        time.sleep(0.1)  # 부저 울림 지속 시간
        ser.write(b'0')  # 부저 OFF 신호 전송
        time.sleep(0.1)  # 부저 울림 간격

# 빈 화면 생성 함수 (새로운 영상 출력용)
def create_empty_canvas(num_windows, window_size, grid_cols):
    grid_rows = (num_windows // grid_cols) + (1 if num_windows % grid_cols != 0 else 0)
    canvas_height = window_size[1] * grid_rows
    canvas_width = window_size[0] * grid_cols
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    return canvas, grid_cols, window_size

# 열화상 카메라 클래스 정의
class FlirLeptonCameraController(object):
    def __init__(self, device_name):
        self.thread_lock = threading.Lock()
        self.t_image_data = None
        self.is_thread_running = False
        self.device_name = device_name
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
        if self.camera is not None:
            self.camera.close()
        if self.capture_thread is not None:
            self.capture_thread.join()

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

    # 열화상 이미지 캡처
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
    if max_temp >= FIRE_THRESHOLD:
        print(f"Fire Detected! Current MaxTemp: {max_temp:.2f}°C")
        trigger_buzzer()

    # 밤과 낮에 따라 입력 이미지 결정
    if is_night:
        frame_thermal_colored = cv2.applyColorMap(frame_thermal.astype(np.uint8), cv2.COLORMAP_HOT)
        input_image = frame_thermal_colored
    else:
        aligned_rgb = cv2.warpPerspective(frame_rgb, H, (frame_thermal.shape[1], frame_thermal.shape[0]))
        gray_rgb = cv2.cvtColor(aligned_rgb, cv2.COLOR_BGR2GRAY)
        gray_rgb_colored = cv2.cvtColor(gray_rgb, cv2.COLOR_GRAY2BGR)
        input_image = gray_rgb_colored

    # YOLOv5 모델로 객체 탐지
    results = model(input_image)

    # 사람 개별 화면 생성용 변수 초기화
    person_videos = []
    for i, result in enumerate(results.xyxy[0]):
        x1, y1, x2, y2, confidence, cls = map(int, result[:6])
        if cls == 0:  # 사람만 감지
            cropped_person = input_image[y1:y2, x1:x2]
            resized_person = cv2.resize(cropped_person, WINDOW_SIZE)
            person_videos.append(resized_person)

            # 바운딩 박스 표시
            cv2.rectangle(input_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 감지된 사람의 개별 영상을 한 화면에 배열
    if person_videos:
        canvas, cols, window_size = create_empty_canvas(len(person_videos), WINDOW_SIZE, grid_cols=3)
        for idx, person_video in enumerate(person_videos):
            row, col = divmod(idx, cols)
            y, x = row * window_size[1], col * window_size[0]
            canvas[y:y + window_size[1], x:x + window_size[0]] = person_video

        # 개별 CCTV 화면 출력
        cv2.imshow("CCTV View - Individuals", canvas)

    # 메인 화면 출력
    cv2.imshow("YOLOv5 Object Detection - Person Only", input_image)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap_rgb.release()
thermal_camera.stop()
cv2.destroyAllWindows()
ser.close()
