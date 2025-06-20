class Validation_Context_Factory:
    def __init__(self, errors, data):
        self.errors = errors
        self.data = data

    def make(self, function, param):
        return Validation_Context(self.errors, self.data, function, param, None)

    def make_with_process_guard(self, function, param, required_process_key):
        return Validation_Context(self.errors, self.data, function, param, required_process_key)

class Validation_Context:
    def __init__(self, errors, data, function, param, required_process_key_or_none):
        self.errors = errors
        self.data = data
        self.function = function
        self.param = param
        self.required_process_key_or_none = required_process_key_or_none

    def do_validation(self):
        if not self.is_param_skipped():
            self.function(self.errors, self.data, self.param)

    def is_param_skipped(self):
        param = self.param
        params = self.data["params"]
        is_skipped = (params[param] is None) or self.is_required_process_disabled() or self.is_alt()
        return is_skipped

    def is_required_process_disabled(self):
        pass

    def is_alt(self):
        value = self.data["params"][self.param]

        if not isinstance(value, str):
            return False

        alt_formats = self.data["specs"][self.param].get("alt_format", [])
        for alt in alt_formats:
            suffix = f".{alt}"
            if value.endswith(suffix):
                return True
        return False
