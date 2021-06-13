

class Replay:

    def __init__(self, k):
        self.k = k
        self.buffer = []

    def add(self, z):
        if len(self.buffer) == self.k:
            self.buffer.pop(0)
            self.buffer.append(z)
        else:
            self.buffer.append(z)
