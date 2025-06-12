def format_list(a_list):
    if len(a_list) == 0:
        return ""
    quoted_list = [f"'{x}'" for x in a_list]
    return quoted_list[-1]
