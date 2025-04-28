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
