###############################################################################
''''''
###############################################################################


import sys
import re


def update_progressbar(i, n):
    if i < 2:
        return
    prog = round(i / (n - 1) * 50)
    sys.stdout.write('\r')
    sys.stdout.write(f"[{prog * '#'}{(50 - prog) * '.'}]")
    sys.stdout.flush()

def remove_brackets(x):
    # Remove brackets from ABS council names:
    return re.sub("[\(\[].*?[\)\]]", "", x).strip()


###############################################################################
###############################################################################
