import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import time
import csv
from datetime import datetime

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
model = YOLO("runs/yoga_pose_detection/weights/best.pt")

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# Wider tolerances for real-world flexibility
POSES = {
    "downdog":   {"left_hip_angle":   (90,  30)},
    "goddess":   {"left_knee_angle":  (90,  25)},
    "tree":      {"right_knee_angle": (60,  25), "left_hip_angle": (170, 20)},
    "warrior2":  {"left_knee_angle":  (90,  25), "left_elbow_angle": (165, 25)},
    "plank":     {"left_elbow_angle": (170, 20), "left_knee_angle":  (170, 20)}
}

def angle_accuracy(actual, target, tolerance):
    diff = abs(actual - target)
    # Softer scoring: accuracy doesn't collapse to 0 immediately
    return max(0, 100 - (diff / tolerance) * 50)

def calculate_pose_accuracy(angles, target_pose):
    required = POSES[target_pose]
    total_accuracy = 0
    count = 0
    for key, (target_angle, tol) in required.items():
        if key in angles:
            acc = angle_accuracy(angles[key], target_angle, tol)
            total_accuracy += acc
            count += 1
    return total_accuracy / count if count > 0 else 0

def log_pose_result(pose, duration, avg_accuracy):
    filepath = "pose_results.csv"
    write_header = False
    try:
        with open(filepath, "r") as f:
            write_header = f.read(1) == ""
    except FileNotFoundError:
        write_header = True

    with open(filepath, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Pose", "Timestamp", "Duration", "Accuracy (%)"])
        writer.writerow([pose, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                         f"{duration}s", f"{avg_accuracy:.2f}%"])

print("Available poses:", list(POSES.keys()))
selected_pose = input("Enter pose to perform: ").strip().lower()
if selected_pose not in POSES:
    print("Invalid pose.")
    exit()

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    hold_start = None
    hold_duration = 3
    pose_logged = False
    accuracy_history = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        results = model.predict(source=frame, conf=0.25, stream=True)
        display_text = f"Pose: {selected_pose}"

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Add padding so landmarks near edges aren't clipped
                pad = 20
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(frame.shape[1], x2 + pad)
                y2 = min(frame.shape[0], y2 + pad)

                person = frame[y1:y2, x1:x2]
                if person.size == 0:
                    continue

                rgb = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)
                angles = {}
                accuracy = 0

                if result.pose_landmarks:
                    lm = result.pose_landmarks.landmark

                    def get(name):
                        p = lm[mp_pose.PoseLandmark[name].value]
                        return int(p.x * person.shape[1]), int(p.y * person.shape[0])

                    try:
                        # LEFT side
                        l_shoulder = get("LEFT_SHOULDER")
                        l_elbow    = get("LEFT_ELBOW")
                        l_wrist    = get("LEFT_WRIST")
                        l_hip      = get("LEFT_HIP")
                        l_knee     = get("LEFT_KNEE")
                        l_ankle    = get("LEFT_ANKLE")

                        # RIGHT side — FIX: use correct right-side landmarks
                        r_hip      = get("RIGHT_HIP")
                        r_knee     = get("RIGHT_KNEE")
                        r_ankle    = get("RIGHT_ANKLE")

                        angles = {
                            "left_elbow_angle":  calculate_angle(l_shoulder, l_elbow, l_wrist),
                            "left_knee_angle":   calculate_angle(l_hip, l_knee, l_ankle),
                            "left_hip_angle":    calculate_angle(l_shoulder, l_hip, l_knee),
                            # FIX: now uses right hip and right ankle
                            "right_knee_angle":  calculate_angle(r_hip, r_knee, r_ankle),
                        }

                        # Debug: print angles to tune targets
                        print({k: f"{v:.1f}°" for k, v in angles.items()})

                        accuracy = calculate_pose_accuracy(angles, selected_pose)
                        mp_drawing.draw_landmarks(person, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    except Exception as e:
                        print("Landmark error:", e)

                if accuracy >= 60:
                    if hold_start is None:
                        hold_start = time.time()
                        accuracy_history = []
                    accuracy_history.append(accuracy)
                    elapsed = time.time() - hold_start
                    if elapsed >= hold_duration and not pose_logged:
                        avg_accuracy = sum(accuracy_history) / len(accuracy_history)
                        display_text = f"Held! Accuracy: {avg_accuracy:.1f}%"
                        log_pose_result(selected_pose, hold_duration, avg_accuracy)
                        pose_logged = True
                    else:
                        remaining = max(0, int(hold_duration - elapsed))
                        display_text = f"Holding... {remaining}s | Acc: {accuracy:.1f}%"
                else:
                    hold_start = None
                    pose_logged = False
                    accuracy_history = []
                    display_text = f"Accuracy: {accuracy:.1f}%"

                color = (0, 255, 0) if accuracy >= 60 else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{selected_pose} {accuracy:.0f}%", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(frame, display_text, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        cv2.imshow("Yoga Pose Accuracy", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()