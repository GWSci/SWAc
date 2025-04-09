def change_input_file(input_file, line_contains, change_to):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    for idx, l in enumerate(lines):
        if line_contains in l:
            lines[idx] = change_to
    with open(input_file, 'w') as f:
        f.writelines(lines)