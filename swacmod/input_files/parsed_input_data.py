from dataclasses import dataclass

@dataclass
class ParsedInputData:
    data: map
    errors: list
    warnings: list

    def print(self, printer=print):
        pass
