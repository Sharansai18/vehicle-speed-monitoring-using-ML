"""Simple centroid tracker: gives each vehicle a persistent ID across frames."""
import math


class Tracker:
    def __init__(self, max_distance=60, max_missed=15):
        self.tracks = {}          # id -> {"center": (cx, cy), "missed": int}
        self.next_id = 1
        self.max_distance = max_distance
        self.max_missed = max_missed

    def update(self, boxes):
        """boxes: list of [x1, y1, x2, y2]. Returns list of [x1, y1, x2, y2, id]."""
        results = []
        used_ids = set()

        for x1, y1, x2, y2 in boxes:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # find nearest existing track not already claimed this frame
            best_id, best_dist = None, self.max_distance
            for tid, t in self.tracks.items():
                if tid in used_ids:
                    continue
                d = math.hypot(cx - t["center"][0], cy - t["center"][1])
                if d < best_dist:
                    best_id, best_dist = tid, d

            if best_id is None:                      # new vehicle
                best_id = self.next_id
                self.next_id += 1

            self.tracks[best_id] = {"center": (cx, cy), "missed": 0}
            used_ids.add(best_id)
            results.append([x1, y1, x2, y2, best_id])

        # age out tracks that were not seen
        for tid in list(self.tracks):
            if tid not in used_ids:
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > self.max_missed:
                    del self.tracks[tid]

        return results
