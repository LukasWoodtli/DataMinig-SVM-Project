#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import sys
import numpy as np

from sklearn.kernel_approximation import RBFSampler
from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.kernel_approximation import SkewedChi2Sampler

WORKING_DIMENSION = 100

kernel = SkewedChi2Sampler(n_components=WORKING_DIMENSION)
#inverse_kernel = RBFSampler(n_components=WORKING_DIMENSION)

DIMENSION = 400  # Dimension of the original data.
CLASSES = (-1, +1)   # The classes that we are trying to predict.

#ETHA = 0.005
ETHA = 0.005

#C = 0.001
C = 0.001

transform_param = 1

def transform(x_original):
    #x_transformed = x_original
    #x_transformed = np.append(x_original, 0.5)
    #x_transformed = x_transformed / np.linalg.norm(x_transformed)
    x_transformed = kernel.fit_transform(x_original)
    #x_transformed = inverse_kernel.fit_transform(x_transformed)
    print x_transformed.size
    assert x_transformed.size == WORKING_DIMENSION
    return x_transformed[0]


def dL_dw(xi, yi, w, j):
    if (yi * np.inner(w, xi)) >= 1:
        return 0
    else:
        return (-1) * yi * xi[j]


def gradient(x,y,w):
    assert w.size == WORKING_DIMENSION
    assert x.size == WORKING_DIMENSION
    w_new = []
    for j, wj in enumerate(w):
        w_new.append(w[j] + C * dL_dw(x, y, w, j))
    return np.array(w_new)


def main(stream):
    w = np.array([0.5 for i in range(WORKING_DIMENSION)])

    assert w.size == WORKING_DIMENSION


    for line in stream[:1000]:
        line = line.strip()
        (label, x_string) = line.split(" ", 1)
        label = int(label)
        x_original = np.fromstring(x_string, sep=' ')
        x = transform(x_original)  # Use our features.

        w = w - ETHA * gradient(x, label, w)
        assert w.size == WORKING_DIMENSION

    assert w.size == WORKING_DIMENSION
    result = ""
    for i in w:
        result += "%f " % i
    return result

if __name__ == "__main__":
   print main(sys.stdin)


