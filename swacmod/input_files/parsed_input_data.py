from dataclasses import dataclass

@dataclass
class ParsedInputData:
    data: map
    errors: list
    warnings: list

    def print(self, printer=print):
        for e in self.errors:
            printer(e)
        for w in self.warnings:
            printer(w)

    def has_errors(self):
        return len(self.errors) > 0

    def update(self, new):
        self.data = new.data
        self.errors += new.errors
        self.warnings += new.warnings
        return self
