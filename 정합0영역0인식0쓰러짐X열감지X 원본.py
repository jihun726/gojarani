import pandas as pd
import cv2
import numpy as np
import torch
import datetime
import warnings
import time

# 경고 메시지 숨기기
warnings.filterwarnings("ignore", category=FutureWarning)

# 엑셀 파일에서 데이터 읽기
df = pd.read_excel('dot_pattern.xlsx')
rgb_points = df[['RGB_X', 'RGB_Y']].values
thermal_points = df[['Thermal_X', 'Thermal_Y']].values

# 호모그래피 매트릭스 계산
H, _ = cv2.findHomography(rgb_points, thermal_points, method=cv2.RANSAC)
print(H)
# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)

# 쓰러짐 감지 기준 설정
FALL_RATIO_THRESHOLD = 1.6  # 비율 임계값
FALL_DETECTION_DURATION = 1  # 감지 지속 시간 (초)

# 두 카메라 캡처 (예: RGB 카메라와 Thermal 카메라)
cap_rgb = cv2.VideoCapture(2)  # RGB 카메라
cap_thermal = cv2.VideoCapture(1)  # 열화상 카메라

fall_start_time = None  # 쓰러짐 감지 시작 시간

# 안전 구역 설정 변수 초기화
safety_zone = []
person_centers = {}
alert_times = {}

# 마우스 콜백 함수: 안전 구역을 다각형으로 정의하거나 초기화하는 역할
def mouse_callback(event, x, y, flags, param):
    global safety_zone
    if event == cv2.EVENT_LBUTTONDOWN:
        safety_zone.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        safety_zone = []  # 우클릭 시 안전 구역 초기화

# 창을 크기 조절 가능한 모드로 설정
cv2.namedWindow('YOLOv5 Object Detection - Person Only', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('YOLOv5 Object Detection - Person Only', mouse_callback)

while True:
    # RGB 이미지 캡처
    ret_rgb, frame_rgb = cap_rgb.read()
    if not ret_rgb:
        print("RGB 카메라에서 프레임을 가져올 수 없습니다.")
        break

    # Thermal 이미지 캡처
    ret_thermal, frame_thermal = cap_thermal.read()
    if not ret_thermal:
        print("열화상 카메라에서 프레임을 가져올 수 없습니다.")
        break

    # 호모그래피 적용 후 이미지 정합
    if H is not None:
        aligned_rgb = cv2.warpPerspective(frame_rgb, H, (frame_thermal.shape[1], frame_thermal.shape[0]))
        result_image = cv2.addWeighted(aligned_rgb, 0.5, frame_thermal, 0.5, 0)

        # YOLOv5 모델을 사용해 객체 탐지
        results = model(result_image)

        # 탐지된 사람 위치 초기화
        new_centers = {}
        fall_detected = False

        # 사람만 필터링하여 탐지 결과 표시
        for i, result in enumerate(results.xyxy[0]):
            x1, y1, x2, y2, confidence, cls = map(int, result[:6])

            if cls == 0:  # 사람만 감지
                width = x2 - x1
                height = y2 - y1
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                person_id = f"person_{i + 1}"
                new_centers[person_id] = (center_x, center_y, width, height)

                # 바운딩 박스와 레이블 표시
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(result_image, f"Person {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # 쓰러짐 감지 로직
                ratio = width / height if height != 0 else 0
                if ratio > FALL_RATIO_THRESHOLD:
                    if fall_start_time is None:
                        fall_start_time = time.time()
                    elif time.time() - fall_start_time >= FALL_DETECTION_DURATION:
                        fall_detected = True
                        cv2.putText(result_image, "Fall Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                else:
                    fall_start_time = None  # 초기화

                # 안전 구역 외부에 있는지 확인
                if len(safety_zone) >= 3:
                    is_inside = cv2.pointPolygonTest(np.array(safety_zone, np.int32), (center_x, center_y), False)
                    if is_inside < 0:
                        current_time = time.time()
                        if person_id not in alert_times or current_time - alert_times[person_id] >= 5:
                            alert_times[person_id] = current_time
                            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            print(f"Dangerous - {person_id} left the zone at {timestamp}")
                            info_text = f"{person_id}: Center({center_x},{center_y}) W:{width} H:{height}"
                            print(info_text)

        # 사람 중심 업데이트
        person_centers = new_centers

        # 안전 구역 표시
        if len(safety_zone) >= 3:
            pts = np.array(safety_zone, np.int32).reshape((-1, 1, 2))
            cv2.polylines(result_image, [pts], isClosed=True, color=(255, 0, 0), thickness=1)

        # 탐지된 사람 수 화면에 표시
        cv2.putText(result_image, f"Count: {len(new_centers)}", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 결과 이미지 출력
        cv2.imshow("YOLOv5 Object Detection - Person Only", result_image)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap_rgb.release()
cap_thermal.release()
cv2.destroyAllWindows()