SPEAK_RANGE = 3.0
REPEAT_EVERY = 4.0
VERY_CLOSE = 0.7


class Announcer:
    def __init__(self):
        self.last_dist = None
        self.last_time = 0
        self.said_stop = False   # NEW: did we already say "Stop"?

    def decide(self, dist, now, speaking):
        """Return (sentence, interrupt).
        sentence = None  -> stay quiet
        interrupt = True -> cut off the current sentence first"""
        if dist >= SPEAK_RANGE:
            self.last_dist = None
            self.said_stop = False
            return None, False

        rounded = round(dist * 2) / 2
        very_close = dist < VERY_CLOSE

        if speaking:
            # NEW: only interrupt for something more urgent
            much_closer = self.last_dist is not None and rounded <= self.last_dist - 1.0
            urgent = (very_close and not self.said_stop) or much_closer
            if not urgent:
                return None, False
            interrupt = True
        else:
            changed = self.last_dist is None or abs(rounded - self.last_dist) >= 0.5
            repeat = now - self.last_time >= REPEAT_EVERY
            if not (changed or repeat):
                return None, False
            interrupt = False

        self.last_dist = rounded
        self.last_time = now
        self.said_stop = very_close
        if very_close:
            return "Stop. Object very close.", interrupt
        return f"Object ahead, {rounded:g} meters.", interrupt