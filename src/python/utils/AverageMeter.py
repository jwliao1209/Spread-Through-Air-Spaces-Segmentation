class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.count = 0

    def update(self, avg, n):
        self.val += (avg * n)
        self.count += n

    def item(self):
        return self.val / self.count if self.count else 0

    def __repr__(self):
        return f"{self.item()}"

    def __str__(self):
        return f"{self.item()}"
