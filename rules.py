from config import SPEAK_RANGE, REPEAT_EVERY, VERY_CLOSE, SIDE_REPEAT

class Announcer:
    def __init__(self):
        self.last_dist = None
        self.last_time = 0
        self.said_stop = False
        self.last_label = None
        self.last_sides = (None, None)   # what we last said about left/right
        self.last_side_time = 0

    def decide(self, dist, now, speaking, label=None, left=None, right=None):
        """Return (sentence, interrupt).
        sentence = None  -> stay quiet
        interrupt = True -> cut off the current sentence first"""
        # NEW: front has priority; if it has nothing to say, sides fill the silence
        message, interrupt = self.front_message(dist, now, speaking, label)
        if message is None:
            message = self.side_message(now, speaking, left, right)
        return message, interrupt

    def front_message(self, dist, now, speaking, label):
        """What's ahead (this is the old decide, unchanged inside)."""
        if dist >= SPEAK_RANGE:
            self.last_dist = None
            self.said_stop = False
            self.last_label = None
            return None, False   # CHANGED: path clear -> nothing ahead (decide() asks the sides)

        rounded = round(dist * 2) / 2
        very_close = dist < VERY_CLOSE

        if speaking:
            # only interrupt for something more urgent
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

    def side_message(self, now, speaking, left, right):
        """Front is quiet: mention what's beside the user, if it changed."""
        sides = (left, right)
        if left is None and right is None:
            self.last_sides = sides # nothing beside me: remember that
            return None
        
        if speaking:
            return None # never interrupt for sides
        
        if sides == self.last_sides and now - self.last_side_time < SIDE_REPEAT:
            return None # same as before, too soon to repeat
        
        self.last_sides = sides
        self.last_side_time = now

        if left == right == "wall":
            return "Walls on both sides."
        parts = []
        if left:
            parts.append(f"{left} on your left")
        if right:
            parts.append(f"{right} on your right")
        return " and ".join(parts).capitalize() + "."