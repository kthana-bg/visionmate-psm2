import os
import sys
import json
import numpy as np
import joblib

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

MODELS_DIR = os.path.join(_ROOT, "models")
RESULTS_DIR = os.path.join(_ROOT, "results")

EYE_MODEL_PATHS = {
    "Custom CNN":     os.path.join(MODELS_DIR, "eye_strain", "custom_cnn.h5"),
    "MobileNetV2":    os.path.join(MODELS_DIR, "eye_strain", "mobilenetv2.h5"),
    "EfficientNetB0": os.path.join(MODELS_DIR, "eye_strain", "efficientnetb0.h5"),
}

POSTURE_MODEL_PATHS = {
    "Custom Residual DNN":       os.path.join(MODELS_DIR, "posture", "custom_residual_dnn.h5"),
    "Random Forest Classifier":  os.path.join(MODELS_DIR, "posture", "mediapipe_model.joblib"),
    "YOLOv8-Pose / MoveNet DNN": os.path.join(MODELS_DIR, "posture", "yolo_movenet_dnn.keras"),
}

POSTURE_SCALER_PATHS = {
    "Custom Residual DNN":       os.path.join(MODELS_DIR, "posture", "scaler_residual_dnn.joblib"),
    "Random Forest Classifier":  None,
    "YOLOv8-Pose / MoveNet DNN": os.path.join(MODELS_DIR, "posture", "scaler_yolo.joblib"),
}

POSTURE_THRESHOLDS = {
    "Custom Residual DNN":       0.3259,
    "Random Forest Classifier":  0.4793,
    "YOLOv8-Pose / MoveNet DNN": 0.8015,
}

RESULTS_PATHS = {
    "Custom CNN":                 os.path.join(RESULTS_DIR, "custom_cnn_results.json"),
    "MobileNetV2":                os.path.join(RESULTS_DIR, "mobilenetv2_results.json"),
    "EfficientNetB0":              os.path.join(RESULTS_DIR, "efficientnetb0_results.json"),
    "Custom Residual DNN":        os.path.join(RESULTS_DIR, "custom_residual_dnn_results.json"),
    "Random Forest Classifier":   os.path.join(RESULTS_DIR, "mediapipe_results.json"),
    "YOLOv8-Pose / MoveNet DNN":  os.path.join(RESULTS_DIR, "yolo_movenet_results.json"),
}

_DEMO_RESULTS = {
    "Custom CNN":                 {"accuracy": 0.9899, "f1_score": 0.9899, "latency_ms": 82.72},
    "MobileNetV2":                {"accuracy": 0.9760, "f1_score": 0.9760, "latency_ms": 83.31},
    "EfficientNetB0":             {"accuracy": 0.9869, "f1_score": 0.9869, "latency_ms": 102.60},
    "Custom Residual DNN":        {"accuracy": 0.7753, "f1_score": 0.7531, "latency_ms": 75.58},
    "Random Forest Classifier":   {"accuracy": 0.8006, "f1_score": 0.7936, "latency_ms": 69.92},
    "YOLOv8-Pose / MoveNet DNN":  {"accuracy": 0.5249, "f1_score": 0.5457, "latency_ms": 78.89},
}


def _load_weights_from_h5(model, weights_path: str):
    try:
        if weights_path.endswith(".keras"):
            model.load_weights(weights_path)
        else:
            model.load_weights(weights_path, by_name=True)
    except Exception as e:
        print(f"  Failed to load weights: {e}")


def _build_custom_cnn():
    import tensorflow as tf
    K = tf.keras

    inp = K.Input(shape=(32, 64, 3), name="eye_input")

    x = K.layers.Conv2D(32, 3, padding="same", name="conv2d")(inp)
    x = K.layers.BatchNormalization(name="batch_normalization")(x)
    x = K.layers.Activation("relu", name="activation")(x)

    x = K.layers.Conv2D(32, 3, padding="same", name="conv2d_1")(x)
    x = K.layers.BatchNormalization(name="batch_normalization_1")(x)
    x = K.layers.Activation("relu", name="activation_1")(x)

    x = K.layers.MaxPooling2D(name="max_pooling2d")(x)
    x = K.layers.Dropout(0.25, name="dropout")(x)

    x = K.layers.Conv2D(64, 3, padding="same", name="conv2d_2")(x)
    x = K.layers.BatchNormalization(name="batch_normalization_2")(x)
    x = K.layers.Activation("relu", name="activation_2")(x)

    x = K.layers.Conv2D(64, 3, padding="same", name="conv2d_3")(x)
    x = K.layers.BatchNormalization(name="batch_normalization_3")(x)
    x = K.layers.Activation("relu", name="activation_3")(x)

    x = K.layers.MaxPooling2D(name="max_pooling2d_1")(x)
    x = K.layers.Dropout(0.25, name="dropout_1")(x)

    x = K.layers.Conv2D(128, 3, padding="same", name="conv2d_4")(x)
    x = K.layers.BatchNormalization(name="batch_normalization_4")(x)
    x = K.layers.Activation("relu", name="activation_4")(x)

    x = K.layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = K.layers.Dropout(0.40, name="dropout_2")(x)
    x = K.layers.Dense(256, activation="relu", name="dense")(x)
    x = K.layers.Dropout(0.40, name="dropout_3")(x)
    out = K.layers.Dense(2, activation="softmax", name="predictions")(x)

    return K.Model(inp, out, name="custom_cnn")


def _build_mobilenetv2():
    import tensorflow as tf
    K = tf.keras

    inp = K.Input(shape=(32, 64, 3), name="eye_input")
    x = K.applications.mobilenet_v2.preprocess_input(inp * 255.0)

    base = K.applications.MobileNetV2(
        input_shape=(32, 64, 3),
        include_top=False,
        weights=None,
    )
    base._name = "mobilenetv2_1.00_224"
    x = base(x)

    x = K.layers.GlobalAveragePooling2D(name="gap")(x)
    x = K.layers.Dense(256, activation="relu", name="dense1")(x)
    x = K.layers.BatchNormalization(name="batch_normalization")(x)
    x = K.layers.Dropout(0.40)(x)
    x = K.layers.Dense(128, activation="relu", name="dense2")(x)
    x = K.layers.Dropout(0.30)(x)
    out = K.layers.Dense(2, activation="softmax", name="predictions")(x)

    return K.Model(inp, out, name="mobilenetv2_model")


def _build_efficientnetb0():
    import tensorflow as tf
    K = tf.keras

    inp = K.Input(shape=(96, 96, 3), name="eye_input")
    x = inp * 255.0

    base = K.applications.EfficientNetB0(
        input_shape=(96, 96, 3),
        include_top=False,
        weights=None,
    )
    base._name = "efficientnetb0"
    x = base(x)

    x = K.layers.GlobalAveragePooling2D(name="gap")(x)
    x = K.layers.Dropout(0.30)(x)
    x = K.layers.Dense(128, activation="relu", name="dense1")(x)
    x = K.layers.BatchNormalization(name="batch_normalization")(x)
    x = K.layers.Dropout(0.40)(x)
    x = K.layers.Dense(64, activation="relu", name="dense2")(x)
    x = K.layers.Dropout(0.30)(x)
    out = K.layers.Dense(2, activation="softmax", name="predictions")(x)

    return K.Model(inp, out, name="efficientnetb0_model")


def _build_custom_residual_dnn():
    import tensorflow as tf
    K = tf.keras

    inp = K.Input(shape=(9,), name="posture_features")

    x = K.layers.Dense(128, name="feature_expansion")(inp)
    x = K.layers.BatchNormalization(name="bn_expansion")(x)
    x = K.layers.Activation("relu", name="act_expansion")(x)

    res1 = K.layers.Dense(128, name="res1_dense1")(x)
    res1 = K.layers.BatchNormalization(name="res1_bn1")(res1)
    res1 = K.layers.Activation("relu", name="res1_act1")(res1)
    res1 = K.layers.Dropout(0.3, name="res1_drop")(res1)
    res1 = K.layers.Dense(128, name="res1_dense2")(res1)
    res1 = K.layers.BatchNormalization(name="res1_bn2")(res1)
    x = K.layers.Add(name="skip_connection_1")([x, res1])
    x = K.layers.Activation("relu", name="res1_out")(x)

    res2 = K.layers.Dense(128, name="res2_dense1")(x)
    res2 = K.layers.BatchNormalization(name="res2_bn1")(res2)
    res2 = K.layers.Activation("relu", name="res2_act1")(res2)
    res2 = K.layers.Dropout(0.3, name="res2_drop")(res2)
    res2 = K.layers.Dense(128, name="res2_dense2")(res2)
    res2 = K.layers.BatchNormalization(name="res2_bn2")(res2)
    x = K.layers.Add(name="skip_connection_2")([x, res2])
    x = K.layers.Activation("relu", name="res2_out")(x)

    x = K.layers.Dense(64, activation="relu", name="pre_classifier_1")(x)
    x = K.layers.Dropout(0.25, name="final_drop_1")(x)
    x = K.layers.Dense(32, activation="relu", name="pre_classifier_2")(x)
    x = K.layers.Dropout(0.2, name="final_drop_2")(x)
    out = K.layers.Dense(2, activation="softmax", name="output")(x)

    return K.Model(inp, out, name="Custom_Residual_DNN")


def _residual_dense_block(x, units, dropout):
    import tensorflow as tf
    K = tf.keras
    shortcut = x
    x = K.layers.Dense(units)(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Activation("relu")(x)
    x = K.layers.Dropout(dropout)(x)
    x = K.layers.Dense(units)(x)
    x = K.layers.BatchNormalization()(x)
    if shortcut.shape[-1] != units:
        shortcut = K.layers.Dense(units, use_bias=False)(shortcut)
    x = K.layers.Add()([x, shortcut])
    x = K.layers.Activation("relu")(x)
    return x


def _build_yolo_movenet_dnn():
    import tensorflow as tf
    K = tf.keras

    inp = K.Input(shape=(7,), name="posture_features")
    x = K.layers.Dense(64)(inp)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.Activation("relu")(x)
    x = _residual_dense_block(x, 64, dropout=0.3)
    x = _residual_dense_block(x, 32, dropout=0.3)
    x = K.layers.Dense(16, activation="relu")(x)
    x = K.layers.Dropout(0.2)(x)
    out = K.layers.Dense(5, activation="softmax", name="output")(x)

    return K.Model(inp, out, name="YOLOv8_MoveNet_DNN_Posture")


_EYE_BUILDERS = {
    "Custom CNN":     _build_custom_cnn,
    "MobileNetV2":    _build_mobilenetv2,
    "EfficientNetB0": _build_efficientnetb0,
}

_POSTURE_KERAS_BUILDERS = {
    "Custom Residual DNN": _build_custom_residual_dnn,
    "YOLOv8-Pose / MoveNet DNN": _build_yolo_movenet_dnn,
}


def _load_eye_model(model_name: str, model_path: str):
    if not model_path or not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None
    builder = _EYE_BUILDERS.get(model_name)
    if builder is None:
        return None
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
        model = builder()
        _load_weights_from_h5(model, model_path)
        print(f"Loaded eye model: {os.path.basename(model_path)}")
        return model
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return None


def _load_posture_model(model_name: str, model_path: str):
    if not model_path or not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None

    if model_name == "Random Forest Classifier":
        try:
            return joblib.load(model_path)
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            return None

    builder = _POSTURE_KERAS_BUILDERS.get(model_name)
    if builder is None:
        return None
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
        model = builder()
        _load_weights_from_h5(model, model_path)
        print(f"Loaded posture model: {os.path.basename(model_path)}")
        return model
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return None


def _load_posture_scaler(model_name: str):
    path = POSTURE_SCALER_PATHS.get(model_name)
    if not path or not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"Failed to load scaler for {model_name}: {e}")
        return None


def load_all_eye_models() -> dict:
    return {
        name: _load_eye_model(name, path)
        for name, path in EYE_MODEL_PATHS.items()
    }


def load_all_posture_models() -> dict:
    return {
        name: _load_posture_model(name, path)
        for name, path in POSTURE_MODEL_PATHS.items()
    }


def load_all_posture_scalers() -> dict:
    return {
        name: _load_posture_scaler(name)
        for name in POSTURE_MODEL_PATHS
    }


def get_posture_threshold(model_name: str) -> float:
    return POSTURE_THRESHOLDS.get(model_name, 0.5)


def load_results(model_name: str) -> dict:
    path = RESULTS_PATHS.get(model_name)
    if path and os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        test = data.get("test", {})
        return {
            "accuracy": test.get("accuracy", data.get("accuracy", 0.0)),
            "f1_score": test.get("f1_score", data.get("f1_score", 0.0)),
            "latency_ms": data.get("latency_ms", 0.0),
        }
    return _DEMO_RESULTS.get(model_name, {"accuracy": 0.0, "f1_score": 0.0, "latency_ms": 0.0})


def load_all_results() -> dict:
    return {name: load_results(name) for name in RESULTS_PATHS}
