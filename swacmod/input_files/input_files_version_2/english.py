def format_list(a_list):
    quoted_list = [f"'{x}'" for x in a_list]
    result = ""
    count = len(quoted_list)
    if count > 0:
        result += quoted_list[0]
    if count > 1:
        result += f" or {quoted_list[-1]}"
    return result
