"""Voice guidance for VisionMate.

Speaks short coaching messages with a pre-trained Piper TTS voice (ONNX, CPU only).
The model files live in assets/voice/:
    visionmate-voice.onnx
    visionmate-voice.onnx.json
    voice_settings.json          (optional: {"length_scale": 1.0})

If Piper or the model is missing, every function here degrades quietly so the
monitoring tab keeps working (the beep still plays).
"""
import base64
import io
import json
import os
import time
import wave

import streamlit as st
import streamlit.components.v1 as components

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOICE_DIR = os.path.join(_ROOT, "assets", "voice")
MODEL_PATH = os.path.join(VOICE_DIR, "visionmate-voice.onnx")
SETTINGS_PATH = os.path.join(VOICE_DIR, "voice_settings.json")

DEFAULT_LENGTH_SCALE = 1.0
DEFAULT_MIN_INTERVAL = 30      # seconds between spoken messages
DEFAULT_VOLUME = 80            # percent

EYE_MESSAGES = [
    "Your eyes look tired. Look at something twenty feet away for twenty seconds.",
    "Time to rest your eyes. Look away from the screen and blink slowly a few times.",
    "Eye strain detected. Please take a short break.",
]
POSTURE_MESSAGES = [
    "You are slouching. Sit up straight and relax your shoulders.",
    "Your head is leaning forward. Bring it back over your shoulders.",
    "Please adjust your posture and keep your back against the chair.",
]
BOTH_MESSAGE = "Both your eyes and your posture need attention. Take a short break and sit up straight."
TEST_MESSAGE = "Voice guidance is on. I will remind you to rest your eyes and sit up straight."


def first_name(full_name):
    """'KEERTHANA D/O BALE MURALI' -> 'Keerthana'. Returns None if there is no usable name."""
    if not full_name:
        return None
    token = str(full_name).strip().split()[0] if str(full_name).strip() else ""
    token = "".join(ch for ch in token if ch.isalpha())
    return token.capitalize() if token else None


def personalise(text, name):
    if not name:
        return text
    return f"{name}, {text[0].lower()}{text[1:]}"


def build_message(eye_status, posture_status, name=None, counter=0):
    """Choose the coaching sentence for the current state. Returns None if nothing needs saying."""
    eye_bad = eye_status == "Strained"
    posture_bad = posture_status == "Slouching"
    if eye_bad and posture_bad:
        text = BOTH_MESSAGE
    elif eye_bad:
        text = EYE_MESSAGES[counter % len(EYE_MESSAGES)]
    elif posture_bad:
        text = POSTURE_MESSAGES[counter % len(POSTURE_MESSAGES)]
    else:
        return None
    return personalise(text, name)


def build_player_html(b64_wav, volume=0.8, delay_ms=600):
    """HTML snippet that plays the clip on the parent page so it survives Streamlit reruns.

    The short delay lets the existing alert beep finish first. If the browser blocks access to
    the parent page, it falls back to playing inside the iframe.
    """
    vol = max(0.0, min(1.0, float(volume)))
    return f"""
    <script>
    (function() {{
      try {{
        var host = window;
        try {{ if (window.parent && window.parent.document) {{ host = window.parent; }} }} catch (e) {{ host = window; }}
        if (host.__vmVoice) {{ try {{ host.__vmVoice.pause(); }} catch (e) {{}} }}
        var a = new host.Audio("data:audio/wav;base64,{b64_wav}");
        a.volume = {vol};
        host.__vmVoice = a;
        setTimeout(function() {{
          a.play().catch(function(err) {{ console.log("VisionMate voice blocked by browser:", err); }});
        }}, {int(delay_ms)});
      }} catch (e) {{ console.log("VisionMate voice error:", e); }}
    }})();
    </script>
    """


# ----------------------------------------------------------------------------
# Model loading and synthesis
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading voice model...")
def _load_voice():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Voice model not found at {MODEL_PATH}")
    from piper import PiperVoice  # imported lazily so the app still starts without Piper
    return PiperVoice.load(MODEL_PATH)


def get_length_scale():
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            return float(json.load(f).get("length_scale", DEFAULT_LENGTH_SCALE))
    except Exception:
        return DEFAULT_LENGTH_SCALE


def voice_error():
    """Returns None if voice guidance is usable, otherwise a short reason string."""
    try:
        _load_voice()
        return None
    except FileNotFoundError:
        return "Voice model files are missing from assets/voice/."
    except ImportError:
        return "The piper-tts package is not installed."
    except Exception as e:  # noqa: BLE001
        return f"Voice model could not be loaded: {e}"


@st.cache_data(show_spinner=False, max_entries=64)
def _synthesize_b64(text, length_scale):
    from piper import SynthesisConfig
    voice = _load_voice()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        voice.synthesize_wav(text, wf, syn_config=SynthesisConfig(length_scale=length_scale))
    data = buf.getvalue()
    with wave.open(io.BytesIO(data), "rb") as wf:
        duration = wf.getnframes() / float(wf.getframerate())
    return base64.b64encode(data).decode("ascii"), duration


# ----------------------------------------------------------------------------
# Streamlit-facing API
# ----------------------------------------------------------------------------
def _setting(key, default):
    return st.session_state.get(key, default)


def speak(text, force_volume=None):
    """Synthesise and play one message. Returns True if it was sent to the browser."""
    if voice_error() is not None:
        return False
    try:
        b64, _ = _synthesize_b64(text, get_length_scale())
    except Exception as e:  # noqa: BLE001
        print(f"[VOICE] synthesis failed: {e}")
        return False
    volume = (force_volume if force_volume is not None else _setting("voice_volume", DEFAULT_VOLUME)) / 100.0
    components.html(build_player_html(b64, volume), height=0, width=0)
    st.session_state["voice_last_text"] = text
    return True


def maybe_speak_alert(eye_status, posture_status, full_name=None):
    """Call this when the beep alert fires. Applies the on/off switch and the minimum interval."""
    if not _setting("voice_enabled", True):
        return False
    now = time.time()
    if now - _setting("voice_last_spoken", 0) < _setting("voice_min_interval", DEFAULT_MIN_INTERVAL):
        return False

    name = first_name(full_name) if _setting("voice_use_name", True) else None
    counter = _setting("voice_msg_counter", 0)
    text = build_message(eye_status, posture_status, name=name, counter=counter)
    if text is None:
        return False

    if speak(text):
        st.session_state["voice_last_spoken"] = now
        st.session_state["voice_msg_counter"] = counter + 1
        return True
    return False


def render_voice_controls():
    """Small settings panel for the Live Monitoring tab."""
    with st.expander("Voice guidance", expanded=False):
        err = voice_error()
        if err:
            st.session_state["voice_enabled"] = False
            st.warning(f"Voice guidance is unavailable. {err} The beep alert still works.")
            return

        col_a, col_b = st.columns(2)
        with col_a:
            st.toggle("Speak coaching alerts", value=True, key="voice_enabled")
        with col_b:
            st.toggle("Use my name", value=True, key="voice_use_name")

        st.slider("Volume", 0, 100, DEFAULT_VOLUME, key="voice_volume")
        st.slider("Minimum seconds between voice messages", 10, 120, DEFAULT_MIN_INTERVAL,
                  key="voice_min_interval")

        if st.button("Test voice", key="voice_test"):
            name = first_name((st.session_state.get("user") or {}).get("full_name")) \
                if _setting("voice_use_name", True) else None
            speak(personalise(TEST_MESSAGE, name))

        last = st.session_state.get("voice_last_text")
        if last:
            st.caption(f"Last message: {last}")
