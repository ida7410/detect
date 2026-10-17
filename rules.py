SPEAK_RANGE = 3.0
REPEAT_EVERY = 4.0
VERY_CLOSE = 0.7


class Announcer:
    def __init__(self):
        self.last_dist = None
        self.last_time = 0
        self.said_stop = False
        self.last_label = None

    def decide(self, dist, now, speaking, label=None):
        """Return (sentence, interrupt).
        sentence = None  -> stay quiet
        interrupt = True -> cut off the current sentence first"""
        if dist >= SPEAK_RANGE:
            self.last_dist = None
            self.said_stop = False
            self.last_label = None
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
            changed = (self.last_dist is None 
                        or abs(rounded - self.last_dist) >= 0.5
                        or label != self.last_label)
            repeat = now - self.last_time >= REPEAT_EVERY
            if not (changed or repeat):
                return None, False
            interrupt = False

        self.last_dist = rounded
        self.last_time = now
        self.said_stop = very_close
        self.last_label = label

        name = label.capitalize() if label else "Object"
        if very_close:
            return f"Stop. {name} very close.", interrupt
        return f"{name} ahead, {rounded:g} meters.", interrupt