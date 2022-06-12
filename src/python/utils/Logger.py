from .FileIO import WriteCsv

__all__ = ["Logger"]

class Logger():
    def __init__(self, print_=False):
        self.records = []
        self.print_ = print_

    def add(self, **inputs):
        self.records.append(inputs)
        if self.print_:
            print(self)
        
    def __str__(self):
        return ', '.join(f"{k}: {v}" for k, v in self.records[-1].items())

    def save(self, path):
        fieldnames = self.records[0].keys()
        WriteCsv(path, fieldnames, self.records)

    def reset(self):
        self.__init__(self.print_)
