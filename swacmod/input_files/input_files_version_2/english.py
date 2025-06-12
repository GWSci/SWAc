def format_list(a_list):
    quoted_list = [f"'{x}'" for x in a_list]
    if len(a_list) == 0:
        return ""
    return quoted_list[-1]
