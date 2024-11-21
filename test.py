import cv2
import torch
import time
import warnings

# 경고 메시지 숨기기
warnings.filterwarnings("ignore", category=FutureWarning)


# YOLOv5 모델 로드 (사전 학습된 weights 사용)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)

# 카메라 캡처 시작
cap = cv2.VideoCapture(2)  # 0은 기본 웹캠을 의미합니다.

# 쓰러짐 감지 및 알람용 변수 초기화
fall_detected_time = None
ALARM_DURATION = 2  # 5초 지속 시 알람

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("Unable to retrieve frame from camera.")
        break

    # YOLOv5 모델로 객체 감지
    results = model(frame)

    # 감지 결과에서 사람만 필터링하여 바운딩 박스와 레이블 표시
    fall_detected = False  # 매 프레임에서 쓰러진 사람을 다시 확인하기 위해 초기화

    for result in results.xyxy[0]:
        x1, y1, x2, y2, confidence, cls = map(int, result[:6])

        # 클래스가 0인 경우 (사람만 감지)
        if cls == 0:
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height

            # 바운딩 박스 그리기 및 자세에 따른 다른 색상 지정
            if aspect_ratio > 1.5:  # 비율 기준으로 누워있는 사람 판단
                label = f"Lying Person {confidence:.2f}"
                color = (0, 0, 255)  # 빨간색 (누워있는 사람)
                fall_detected = True  # 쓰러진 사람 감지 표시
            else:
                label = f"Person {confidence:.2f}"
                color = (0, 255, 0)  # 초록색 (서있는 사람)

            # 바운딩 박스와 레이블 표시
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 쓰러진 사람이 감지된 경우
    if fall_detected:
        # 쓰러진 상태 감지 시간 설정
        if fall_detected_time is None:
            fall_detected_time = time.time()  # 쓰러진 상태 시작 시간 기록
        # 쓰러짐이 5초 이상 지속되면 알람
        elif time.time() - fall_detected_time >= ALARM_DURATION:
            cv2.putText(frame, "ALARM: Fall Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            print("Fall Detection Alert!")  # 알람 출력 (추가로 부저를 연결하면 해당 부분에 코드 삽입 가능)
    else:
        # 쓰러진 사람이 감지되지 않으면 시간 초기화
        fall_detected_time = None

    # 결과 프레임 출력
    cv2.imshow("YOLOv5 Person Detection", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
