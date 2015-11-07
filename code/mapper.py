#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import sys
import numpy as np

from sklearn.kernel_approximation import  RBFSampler
from sklearn.linear_model import  SGDClassifier


DIMENSION = 400  # Dimension of the original data.
CLASSES = (-1, +1)   # The classes that we are trying to predict.



rbfSampler = RBFSampler(n_components=100)
fitter_array = np.ndarray(DIMENSION)
fitter_array.fill(0)
rbfSampler.fit(fitter_array)

sgd = SGDClassifier()


def transform(x_original):
    return rbfSampler.transform(x_original)

def main(stream):
    all_data = []
    all_label = []

    for line in stream:
        line = line.strip()
        (label, x_string) = line.split(" ", 1)
        label = int(label)
        all_label.append(label)
        x_original = np.fromstring(x_string, sep=' ')
        x = transform(x_original)  # Use our features.
        all_data.append(x)

    all_label = np.array(all_label)
    all_data = np.array(all_data)
    print all_data
    sgd.fit(all_data, all_label)

    print sgd._coeff



if __name__ == "__main__":
    main(sys.stdin)
