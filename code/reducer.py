#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import logging
import sys
import numpy as np

def main(stream):
    number_of_elemts = 0
    w_new = None
    for line in stream.splitlines():
        line = line.strip()
        line = np.fromstring(line, sep=' ')
        if w_new:
            w_new = np.add(w_new, line)
        else:
            w_new = line

        number_of_elemts += 1

    w = np.divide(w_new, number_of_elemts)

    result = ""
    for i in w:
        result += "%f " % i
    return result

if __name__ == "__main__":
    print main(sys.stdin)

