def format_list(a_list):
    quoted_list = [f"'{x}'" for x in a_list]
    result = ""
    if len(a_list) > 0:
        result += quoted_list[0]
    return result
