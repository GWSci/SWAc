def format_list(a_list):
    quoted_list = [f"'{x}'" for x in a_list]
    result = ""
    count = len(quoted_list)
    if count > 0:
        result += quoted_list[0]
    for x in quoted_list[1:-1]:
        result += f", {x}"
    if count > 1:
        result += f" or {quoted_list[-1]}"
    return result

def convert_type_to_string(spec_type):
    word = convert_type_to_english(spec_type[0])

    article = find_article(word)

    if spec_type[0] == set:
        suffix = f" of {convert_type_to_english(spec_type[1])}s"
    elif spec_type[0] == dict and len(spec_type) == 2:
        suffix = f" mapping integers to {convert_type_to_english(spec_type[1])}s"
    elif spec_type[0] == dict and len(spec_type) == 3:
        suffix = (f" mapping integers"
            + f" to {convert_type_to_english(spec_type[1])}" 
            + f" of {convert_type_to_english(spec_type[2])}s")
    elif spec_type[0] == list and len(spec_type) == 3:
        suffix = (f" of {convert_type_to_english(spec_type[1])}" 
            + f" of {convert_type_to_english(spec_type[2])}s")
    else:
        suffix = ""

    return f"{article} {word}{suffix}"

def convert_type_to_english(t):
    type_to_english = {
        str: "string",
        int: "integer",
        bool: "boolean",
        dict: "dictionary"
    }
    return type_to_english.get(t, t.__name__)

def find_article(word):
    if word[0] in ["a", "e", "i", "o", "u"]:
        return "an"
    return "a"
