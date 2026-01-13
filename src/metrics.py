# metrics.py
import time
from collections import defaultdict

class AgentMetrics:
    def __init__(self):
        self.times = defaultdict(list)
        self.calls = defaultdict(int)

    def start_timer(self, name: str):
        self.times[name].append(time.time())

    def end_timer(self, name: str):
        if self.times[name]:
            elapsed = time.time() - self.times[name].pop()
            self.times[name][-1] = elapsed  # хранить длительность

    def increment(self, name: str):
        self.calls[name] += 1

    def report(self):
        print("\n=== METRICS AGENT ===")
        for k in self.times:
            if self.times[k]:
                avg = sum(self.times[k]) / len(self.times[k])
                print(f"{k}: {len(self.times[k])} calls, avg: {avg:.3f}s")
        for k, v in self.calls.items():
            print(f"{k}: {v} calls")
