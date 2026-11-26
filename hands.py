
import cv2
import mediapipe as mp
import numpy as np
import time
import urllib.request
from pathlib import Path

# Download the pretrained hand detection model
MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)

if not MODEL_PATH.exists():
    print("Downloading hand tracking model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Initialize MediaPipe
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

# Hand skeleton connections
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

FINGERS = {
    "Thumb":  (2, 3, 4),
    "Index":  (5, 6, 7),
    "Middle": (9, 10, 11),
    "Ring":   (13, 14, 15),
    "Pinky":  (17, 18, 19)
}


# Calculate the angle between three points
def calculate_angle(a, b, c):

    vector1 = a - b
    vector2 = c - b

    magnitude = (
        np.linalg.norm(vector1)
        * np.linalg.norm(vector2)
    )

    if magnitude < 1e-8:
        return 180.0

    cosine = np.dot(vector1, vector2) / magnitude
    cosine = np.clip(cosine, -1.0, 1.0)

    return np.degrees(np.arccos(cosine))


# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

smoothed = None
alpha = 0.3
last_timestamp = -1

try:
    with mp.tasks.vision.HandLandmarker.create_from_options(
        options
    ) as detector:

        while True:

            success, frame = cap.read()

            if not success:
                break

            # Mirror the webcam image
            frame = cv2.flip(frame, 1)

            h, w, _ = frame.shape

            # Convert BGR image to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp = max(
                last_timestamp + 1,
                time.monotonic_ns() // 1_000_000
            )

            last_timestamp = timestamp

            # Detect hand landmarks
            result = detector.detect_for_video(
                image, timestamp
            )

            if result.hand_landmarks:

                landmarks = result.hand_landmarks[0]
                world = result.hand_world_landmarks[0]

                # Convert landmarks into NumPy arrays
                points_3d = np.array([
                    [p.x, p.y, p.z] for p in world
                ])

                points_2d = [
                    (int(p.x * w), int(p.y * h))
                    for p in landmarks
                ]

                # Draw skeleton connections
                for a, b in CONNECTIONS:
                    cv2.line(
                        frame,
                        points_2d[a],
                        points_2d[b],
                        (0, 255, 0),
                        2
                    )

                # Draw individual joints
                for point in points_2d:
                    cv2.circle(
                        frame, point, 5,
                        (0, 0, 255), -1
                    )

                # Calculate bending of each finger
                curls = []

                for name, (a, b, c) in FINGERS.items():

                    angle = calculate_angle(
                        points_3d[a],
                        points_3d[b],
                        points_3d[c]
                    )

                    bend = np.clip(
                        (180 - angle) / 100,
                        0, 1
                    )

                    curls.append(bend * 100)

                curls = np.array(curls)

                # Smooth values to reduce jitter
                if smoothed is None:
                    smoothed = curls
                else:
                    smoothed = (
                        alpha * curls
                        + (1 - alpha) * smoothed
                    )

                # Display finger bending percentages
                for i, name in enumerate(FINGERS):

                    value = int(smoothed[i])
                    y = 60 + i * 55

                    cv2.putText(
                        frame,
                        f"{name}: {value}%",
                        (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                    cv2.rectangle(
                        frame,
                        (20, y + 8),
                        (220, y + 22),
                        (255, 255, 255),
                        1
                    )

                    cv2.rectangle(
                        frame,
                        (20, y + 8),
                        (20 + value * 2, y + 22),
                        (0, 255, 0),
                        -1
                    )

                # Five continuous finger movement values
                finger_values = smoothed.astype(int)

                print(finger_values, end="\r")

            else:
                smoothed = None

                cv2.putText(
                    frame,
                    "No hand detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            cv2.imshow("Real-time Hand Tracking", frame)

            # Press Q to exit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()
