#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from math import sqrt

scaler = StandardScaler()


DIMENSION = 400  # Dimension of the original data.
CLASSES = (-1, +1)   # The classes that we are trying to predict.

#ETHA = 0.005
ETHA = 1.0

#C = 0.001


G_diag = np.full(DIMENSION + 1, 0.0, dtype=np.float128)

def transform(x_original):
    x_transformed = np.append(x_original, 0.5)
    #x_transformed = x_transformed / np.linalg.norm(x_transformed)
    x_transformed  = scaler.fit_transform(x_transformed)
    assert x_transformed.size == 401
    return x_transformed


def dL_dw(xi, yi, w, j):
    if (yi * np.inner(w, xi)) >= 1:
        return 0
    else:
        return (-1) * yi * xi[j]


def gradient(x,y,w):
    assert w.size == 401
    assert x.size == 401
    global G_diag


    for j, wj in enumerate(w):
        g_tj = dL_dw(x, y, w, j)
        G_diag[j] = G_diag[j] + g_tj**2
        if G_diag[j] == 0:
            w[j] = wj
        else:
            w[j] = wj - ETHA / sqrt(G_diag[j]) * g_tj
    #result = np.diag(result)
    assert w.size == 401
    return w


def main(stream):
    w = np.array([0.5 for i in range(DIMENSION + 1)])

    assert w.size == 401


    for line in stream:
        line = line.strip()
        (label, x_string) = line.split(" ", 1)
        label = int(label)
        x_original = np.fromstring(x_string, sep=' ')
        x = transform(x_original)  # Use our features.

        w = gradient(x, label, w)


        assert w.size == 401

    print w
    assert w.size == 401
    result = ""
    for i in w:
        result += "%f " % i
    return result

if __name__ == "__main__":
   print main(sys.stdin)


