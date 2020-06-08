import numpy as np

class EWMA():
    "Exponentially weighted moving average."

    def __init__(self, rate=0.8, initial_value = 0.0):
        self.rate = rate
        self.value = initial_value
        self.initial_value = initial_value

    def update(self, x):
        if self.value is None:
            self.value = x
        else:
            self.value += self.rate*(x - self.value)

    def __call__(self):
        return self.value

    def reset(self):
        self.value = self.initial_value

class SimpleMovingAveraging():
    "Simple moving averaging."

    def __init__(self, n=4):
        self.values = []
        self.n = n

    def update(self, x):
        if x is None: return
        if len(self.values) > self.n:
            del self.values[0]
        self.values.append(x)

    def __call__(self):
        if not self.values: return 0
        return np.mean(self.values)

    def reset(self):
        self.values = []

class Averaging():
    "Simple averaging."

    def __init__(self):
        self.count = 0.0
        self.value = 0.0

    def update(self, x):
        if x is None: return
        self.count += 1.0
        self.value += float(x)

    def __call__(self):
        if self.count == 0: return 0
        return self.value / self.count

    def reset(self):
        self.count = 0.0
        self.value = 0.0
