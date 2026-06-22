import streamlit as st
import time
import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.frame_processor import FrameResult, load_mediapipe_landmarkers
from utils.beep_alert import BeepAlert, play_beep
from database.db_manager import save_health_metric


def _get_rtc_configuration():
    try:
        from twilio.rest import Client
        sid = os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
        token = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
        if sid and token:
            client = Client(sid, token)
            token_obj = client.tokens.create()
            return {"iceServers": token_obj.ice_servers}
    except Exception as e:
        print(f"Twilio TURN fetch failed, falling back to STUN: {e}")
    return {
        "iceServers": [
            {"urls": "stun:stun.l.google.com:19302"},
            {"urls": "stun:stun1.l.google.com:19302"},
        ]
    }


def _get_status_color(status: str, good_value: str = "Normal") -> str:
    return "#2ecc71" if status == good_value else "#e74c3c"


def _get_health_color(score: float) -> str:
    if score >= 75:
        return "#2ecc71"
    elif score >= 50:
        return "#f39c12"
    return "#e74c3c"


def _get_coach_advice(eye_status: str, posture_status: str, face_detected: bool) -> tuple:
    if not face_detected:
        return (
            "No face detected. Please position yourself in front of the camera "
            "so eye and posture monitoring can resume.",
            "#7f8c8d",
        )
    eye_bad = eye_status == "Strained"
    posture_bad = posture_status == "Slouching"

    if eye_bad and posture_bad:
        return (
            "Eye strain and bad posture detected. Look away from the screen, "
            "blink a few times, and sit up straight with your head aligned over your shoulders.",
            "#e74c3c",
        )
    if eye_bad:
        return (
            "Eye strain detected. Look at something 20 feet away for 20 seconds "
            "and blink a few times to refresh your eyes.",
            "#e67e22",
        )
    if posture_bad:
        return (
            "Bad posture detected. Sit up straight, pull your shoulders back, "
            "and align your head over your shoulders rather than leaning forward.",
            "#e67e22",
        )
    return ("Eyes and posture look good. Keep up the healthy habit.", "#2ecc71")


def _metric_card(label: str, value: str, color: str, sub_text: str = ""):
    sub_html = (
        f"<p style='font-size:12px;color:#aaa;margin:2px 0 0 0;'>{sub_text}</p>"
        if sub_text else ""
    )
    st.markdown(
        f"""
        <div style="
            background:#1e2130;
            border-left:4px solid {color};
            border-radius:8px;
            padding:14px 16px;
            margin-bottom:10px;
        ">
            <p style="font-size:11px;color:#aaa;margin:0 0 4px 0;
                      text-transform:uppercase;letter-spacing:1px;">{label}</p>
            <p style="font-size:24px;font-weight:bold;color:{color};margin:0;">{value}</p>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _coach_card(message: str, color: str):
    st.markdown(
        f"""
        <div style="
            background:#1e2130;
            border-left:4px solid {color};
            border-radius:8px;
            padding:14px 16px;
            margin-bottom:10px;
        ">
            <p style="font-size:11px;color:#aaa;margin:0 0 6px 0;
                      text-transform:uppercase;letter-spacing:1px;">Live Coach</p>
            <p style="font-size:14px;font-weight:500;color:{color};margin:0;line-height:1.5;">{message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_metrics_panel(result: FrameResult, eye_model_name: str, posture_model_name: str):
    st.markdown(
        "<p style='font-size:13px;font-weight:600;color:#ccc;"
        "text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;'>"
        "Live Analysis</p>",
        unsafe_allow_html=True,
    )
    _metric_card(
        "Eye Status",
        result.eye_status,
        _get_status_color(result.eye_status, "Normal"),
        f"EAR: {result.ear_value:.3f}  |  Confidence: {result.eye_confidence:.2f}",
    )
    _metric_card(
        "Posture Status",
        result.posture_status,
        _get_status_color(result.posture_status, "Good"),
        f"Neck angle: {result.posture_angle:.1f} deg  |  Confidence: {result.posture_confidence:.2f}",
    )
    advice, advice_color = _get_coach_advice(result.eye_status, result.posture_status, result.face_detected)
    _coach_card(advice, advice_color)
    st.markdown(
        f"""
        <div style="font-size:11px;color:#aaa;margin-top:10px;
                    background:#1e2130;border-radius:6px;padding:10px;line-height:1.8;">
            <b>Eye model</b>: {eye_model_name}<br>
            <b>Posture model</b>: {posture_model_name}<br>
            Eye latency: {result.eye_latency_ms:.1f} ms<br>
            Posture latency: {result.posture_latency_ms:.1f} ms
        </div>
        """,
        unsafe_allow_html=True,
    )
    face_color = "#2ecc71" if result.face_detected else "#e74c3c"
    face_text = "Face Detected" if result.face_detected else "No Face Detected"
    st.markdown(
        f"""
        <div style="margin-top:8px;padding:8px 12px;
                    background:{face_color}22;border-radius:6px;
                    border:1px solid {face_color};
                    color:{face_color};font-size:12px;font-weight:600;">
            {face_text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_monitoring_tab(
    processor,
    eye_model_name: str,
    posture_model_name: str,
    user_id: int,
):
    st.header("Live Monitoring")

    try:
        from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
        from utils.frame_processor import VisionMateTransformer, WEBRTC_AVAILABLE
        if not WEBRTC_AVAILABLE:
            raise ImportError("VisionMateTransformer unavailable")
    except ImportError as err:
        st.error(f"streamlit-webrtc not available: {err}")
        return

    if "beep_alert" not in st.session_state:
        st.session_state["beep_alert"] = BeepAlert()
    beep_alert = st.session_state["beep_alert"]

    col_start, col_stop = st.columns(2)
    with col_start:
        if st.button("Start Session", use_container_width=True, key="mon_start"):
            st.session_state["monitoring_active"] = True
            st.session_state["session_start"] = time.time()
            beep_alert.reset_all()
    with col_stop:
        if st.button("Stop Session", use_container_width=True, key="mon_stop"):
            st.session_state["monitoring_active"] = False
            beep_alert.reset_all()

    st.divider()

    if not st.session_state.get("monitoring_active", False):
        st.info("Click Start Session to begin live monitoring.")
        return

    if "mp_landmarkers" not in st.session_state:
        with st.spinner("Loading MediaPipe landmarkers..."):
            st.session_state["mp_landmarkers"] = load_mediapipe_landmarkers()
    face_lm, pose_lm = st.session_state["mp_landmarkers"]

    if "vm_eye_models" not in st.session_state or "vm_posture_models" not in st.session_state:
        from utils.model_loader import load_all_eye_models, load_all_posture_models, load_all_posture_scalers
        with st.spinner("Loading AI models..."):
            st.session_state["vm_eye_models"] = load_all_eye_models()
            st.session_state["vm_posture_models"] = load_all_posture_models()
            st.session_state["vm_posture_scalers"] = load_all_posture_scalers()

    eye_models = st.session_state["vm_eye_models"]
    posture_models = st.session_state["vm_posture_models"]
    posture_scalers = st.session_state["vm_posture_scalers"]

    eye_model = eye_models.get(eye_model_name)
    posture_model = posture_models.get(posture_model_name)
    posture_scaler = posture_scalers.get(posture_model_name)

    if eye_model is None:
        st.warning(f"Eye model '{eye_model_name}' failed to load.")
    if posture_model is None:
        st.warning(f"Posture model '{posture_model_name}' failed to load.")

    feed_col, analysis_col = st.columns([3, 2])

    with feed_col:
        rtc_config = RTCConfiguration(_get_rtc_configuration())

        ctx = webrtc_streamer(
            key="visionmate-live",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            video_transformer_factory=VisionMateTransformer,
            media_stream_constraints={
                "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
                "audio": False,
            },
            async_processing=True,
        )

        if ctx.video_transformer:
            ctx.video_transformer.update_models(
                face_lm, pose_lm,
                eye_model, eye_model_name,
                posture_model, posture_model_name,
                posture_scaler,
            )

        if "session_start" in st.session_state:
            elapsed = int(time.time() - st.session_state["session_start"])
            mins, secs = divmod(elapsed, 60)
            hrs, mins = divmod(mins, 60)
            timer_str = (
                f"{hrs:02d}:{mins:02d}:{secs:02d}" if hrs > 0
                else f"{mins:02d}:{secs:02d}"
            )
            st.caption(f"Session duration: {timer_str}")

    with analysis_col:
        result = ctx.video_transformer.get_result() if ctx and ctx.video_transformer else FrameResult()
        _render_metrics_panel(result, eye_model_name, posture_model_name)

        eye_alert = beep_alert.update_condition("eye_strain", result.eye_status == "Strained")
        posture_alert = beep_alert.update_condition("slouching", result.posture_status == "Slouching")

        if eye_alert or posture_alert:
            play_beep()
            reason = "Eye strain" if eye_alert else "Bad posture"

        last_save = st.session_state.get("last_metric_save", 0)
        if time.time() - last_save >= 5 and result.face_detected:
            save_health_metric(
                user_id=user_id,
                eye_status=result.eye_status,
                ear_value=result.ear_value,
                posture_status=result.posture_status,
                posture_angle=result.posture_angle,
                health_score=result.health_score,
                active_eye_model=eye_model_name,
                active_posture_model=posture_model_name,
            )
            st.session_state["last_metric_save"] = time.time()

    if st.session_state.get("monitoring_active", False):
        time.sleep(2)
        st.rerun()
