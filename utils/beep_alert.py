import time
import streamlit.components.v1 as components

TRIGGER_DELAY_SECONDS = 10
COOLDOWN_SECONDS = 5


class BeepAlert:

    def __init__(self):
        self._condition_start = {}
        self._last_trigger_time = {}

    def update_condition(self, condition_key: str, is_active: bool) -> bool:
        now = time.time()

        if not is_active:
            self._condition_start.pop(condition_key, None)
            return False

        if condition_key not in self._condition_start:
            self._condition_start[condition_key] = now

        elapsed = now - self._condition_start[condition_key]
        if elapsed < TRIGGER_DELAY_SECONDS:
            return False

        last_trigger = self._last_trigger_time.get(condition_key, 0)
        if now - last_trigger < COOLDOWN_SECONDS:
            return False

        self._last_trigger_time[condition_key] = now
        return True

    def reset_condition(self, condition_key: str):
        self._condition_start.pop(condition_key, None)

    def reset_all(self):
        self._condition_start.clear()
        self._last_trigger_time.clear()


def play_beep():
    components.html(
        """
        <script>
        try {
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = "sine";
            osc.frequency.value = 880;
            gain.gain.value = 0.25;
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start();
            osc.stop(ctx.currentTime + 0.4);
        } catch (e) {}
        </script>
        """,
        height=0,
        width=0,
    )
