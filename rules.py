SPEAK_RANGE = 3.0 # only talk about things closer than this (meters)
REPEAT_EVERY = 4.0 # repeat a warning every N seconds if it's still there
VERY_CLOSE = 0.7 # closer than this = "STOP"

class Announcer:
    def __init__(self):
        self.last_dist = None
        self.last_time = 0

    def decide(self, dist, now, speaking):
        """return the sentence to say (None to stay quiet)"""
        if dist >= SPEAK_RANGE:
            self.last_dist = None
            return None
        if speaking:
            return None

        rounded = round(dist * 2) / 2
        changed = self.last_dist is None or abs(rounded - self.last_dist) >= 0.5
        repeat = now - self.last_time >= REPEAT_EVERY
        if not (changed or repeat):
            return None

        self.last_dist = rounded
        self.last_time = now
        if dist < VERY_CLOSE:
            return "STOP. Object very close."
        return f"Object ahead, {rounded:g} meters."