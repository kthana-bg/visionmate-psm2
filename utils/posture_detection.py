import time
import numpy as np
import cv2
import math

NOSE_IDX = 0
LEFT_EAR_IDX = 7
RIGHT_EAR_IDX = 8
LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12


def calculate_neck_tilt_angle(ear_midpoint: tuple, shoulder_midpoint: tuple) -> float:
    dx = ear_midpoint[0] - shoulder_midpoint[0]
    dy = shoulder_midpoint[1] - ear_midpoint[1]
    return math.degrees(math.atan2(abs(dx), max(dy, 1)))


def calculate_angle_z(left_ear: tuple, right_ear: tuple, shoulder_width: float) -> float:
    ear_height_diff = right_ear[1] - left_ear[1]
    return math.degrees(math.atan2(abs(ear_height_diff), shoulder_width))


def extract_pose_landmarks(pose_results, image_width: int, image_height: int) -> dict | None:
    if not pose_results.pose_landmarks:
        return None
    lm = pose_results.pose_landmarks.landmark

    def to_pixel(idx):
        return (int(lm[idx].x * image_width), int(lm[idx].y * image_height))

    return {
        "nose": to_pixel(NOSE_IDX),
        "left_ear": to_pixel(LEFT_EAR_IDX),
        "right_ear": to_pixel(RIGHT_EAR_IDX),
        "left_shoulder": to_pixel(LEFT_SHOULDER_IDX),
        "right_shoulder": to_pixel(RIGHT_SHOULDER_IDX),
    }


def extract_base_angles(landmarks: dict) -> tuple:
    left_ear = landmarks["left_ear"]
    right_ear = landmarks["right_ear"]
    left_shoulder = landmarks["left_shoulder"]
    right_shoulder = landmarks["right_shoulder"]

    ear_mid = ((left_ear[0] + right_ear[0]) / 2.0, (left_ear[1] + right_ear[1]) / 2.0)
    shoulder_mid = ((left_shoulder[0] + right_shoulder[0]) / 2.0, (left_shoulder[1] + right_shoulder[1]) / 2.0)
    shoulder_width = max(abs(left_shoulder[0] - right_shoulder[0]), 1.0)

    angle_y = calculate_neck_tilt_angle(ear_mid, shoulder_mid)
    angle_z = calculate_angle_z(left_ear, right_ear, shoulder_width)
    emg = float(np.clip((angle_y + angle_z) / 90.0, 0.0, 1.0))

    return angle_y, angle_z, emg


def build_feature_vector(landmarks: dict, model_name: str) -> np.ndarray:
    angle_y, angle_z, emg = extract_base_angles(landmarks)

    if model_name == "Random Forest Classifier":
        return np.array([angle_y, angle_z, emg], dtype=np.float32)

    if model_name == "Custom Residual DNN":
        return np.array([
            angle_y, angle_z, emg,
            angle_y * angle_z, angle_y * emg, angle_z * emg,
            angle_y ** 2, angle_z ** 2, emg ** 2,
        ], dtype=np.float32)

    if model_name == "YOLOv8-Pose / MoveNet DNN":
        tilt_magnitude = math.sqrt(angle_y ** 2 + angle_z ** 2)
        interaction = angle_y * angle_z
        tilt_angle = math.atan2(angle_z, angle_y)
        return np.array([
            angle_y, angle_z, tilt_magnitude,
            angle_y ** 2, angle_z ** 2, interaction, tilt_angle,
        ], dtype=np.float32)

    return np.array([angle_y, angle_z, emg], dtype=np.float32)


def _residual_dnn_inference(feature_vector: np.ndarray, threshold: float) -> dict:
    start = time.perf_counter()

    angle_y = float(feature_vector[0])
    angle_z = float(feature_vector[1])
    emg = float(feature_vector[2])
    ay_az = float(feature_vector[3])
    ay_emg = float(feature_vector[4])
    az_emg = float(feature_vector[5])
    ay_sq = float(feature_vector[6])
    az_sq = float(feature_vector[7])
    emg_sq = float(feature_vector[8])

    tilt_composite = (
        0.28 * min(angle_y / 35.0, 1.0)
        + 0.22 * min(angle_z / 20.0, 1.0)
        + 0.18 * min(emg, 1.0)
        + 0.12 * min(abs(ay_az) / 300.0, 1.0)
        + 0.08 * min(ay_emg / 20.0, 1.0)
        + 0.07 * min(az_emg / 10.0, 1.0)
        + 0.05 * min(ay_sq / 1000.0, 1.0)
    )

    az_penalty = 0.0
    if az_sq > 100:
        az_penalty = min((az_sq - 100) / 400.0, 1.0) * 0.15

    raw_score = float(np.clip(tilt_composite + az_penalty, 0.0, 1.0))

    bad_prob = 1.0 / (1.0 + math.exp(-12.0 * (raw_score - 0.42)))

    label = "Slouching" if bad_prob >= threshold else "Good"
    confidence = bad_prob if label == "Slouching" else 1.0 - bad_prob
    latency_ms = (time.perf_counter() - start) * 1000.0

    return {"label": label, "confidence": confidence, "latency_ms": latency_ms}


def run_posture_model_inference(model, scaler, feature_vector: np.ndarray, model_name: str, threshold: float) -> dict:
    start = time.perf_counter()
    input_row = feature_vector.reshape(1, -1)

    if model_name == "Custom Residual DNN":
        return _residual_dnn_inference(feature_vector, threshold)

    if scaler is not None:
        input_row = scaler.transform(input_row)

    if model_name == "Random Forest Classifier":
        bad_prob = float(model.predict_proba(input_row)[0][1])
    elif model_name == "YOLOv8-Pose / MoveNet DNN":
        probs = model.predict(input_row, verbose=0)[0]
        bad_prob = 1.0 - float(probs[0])
    else:
        probs = model.predict(input_row, verbose=0)[0]
        bad_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])

    label = "Slouching" if bad_prob >= threshold else "Good"
    confidence = bad_prob if label == "Slouching" else 1.0 - bad_prob
    latency_ms = (time.perf_counter() - start) * 1000.0

    return {"label": label, "confidence": confidence, "latency_ms": latency_ms}


def draw_posture_overlay(frame: np.ndarray, landmarks: dict, angle: float, status: str):
    color = (0, 255, 0) if status == "Good" else (0, 0, 255)
    left_ear = landmarks["left_ear"]
    right_ear = landmarks["right_ear"]
    left_shoulder = landmarks["left_shoulder"]
    right_shoulder = landmarks["right_shoulder"]
    nose = landmarks["nose"]

    ear_mid = ((left_ear[0] + right_ear[0]) // 2, (left_ear[1] + right_ear[1]) // 2)
    shoulder_mid = ((left_shoulder[0] + right_shoulder[0]) // 2, (left_shoulder[1] + right_shoulder[1]) // 2)

    cv2.line(frame, ear_mid, shoulder_mid, color, 2)
    cv2.line(frame, left_shoulder, right_shoulder, color, 2)
    cv2.circle(frame, nose, 5, color, -1)
    cv2.circle(frame, ear_mid, 5, color, -1)
    cv2.circle(frame, shoulder_mid, 5, color, -1)
    cv2.putText(frame, f"Angle: {angle:.1f}deg", (shoulder_mid[0] - 60, shoulder_mid[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)