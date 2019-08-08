import sys


def parse(names):
    name_dict = {name: None for name in names}

    for i in range(1, len(sys.argv)):
        if sys.argv[i] in name_dict:
            if len(sys.argv) > i + 1:
                name_dict[sys.argv[i]] = sys.argv[i + 1]
            else:
                print("Missing argument value")

    return name_dict
