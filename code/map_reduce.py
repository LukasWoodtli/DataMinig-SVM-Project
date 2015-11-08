import mapper
import reducer
import os
import evaluate

from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np


# Components:  300
# Gamma:  0.1
# Alpha: : 0.01
# 0.696660
# ====================
# Components:  200
# Gamma:  0.1
# Alpha: : 0.05
# 0.726270


DIMENSION = 400  # Dimension of the original data.

WORKING_DIR = os.path.dirname(__file__)

IN_FILE = os.path.join(WORKING_DIR, "..", "data", "xaj")

EVALUATE_SCRIPT = os.path.join(WORKING_DIR, "evaluate.py")

for components in [200]:
    for g in [0.01, 0.05, 0.2]:
        for a in [0.3, 0.5, 0.9, 2]:

            print "=" * 20
            print "Components: ", components
            print "Gamma: ", g
            print "Alpha: :", a

            mapper.rbf_feature = RBFSampler(gamma=g, n_components=components, random_state=42)
            fitter = np.ndarray(shape=(DIMENSION))
            fitter.fill(0)
            mapper.rbf_feature.fit(fitter)

            mapper.clf = SVC(kernel='linear') #SGDClassifier(loss='hinge', penalty ='l2', alpha=a,  n_jobs=-1, random_state=43)

            str = open(IN_FILE).read()
            stream = mapper.main(str)
            # #print stream
            #
            output = reducer.main(stream)
            #
            WEIGHTS_PATH = os.path.join(WORKING_DIR, "weights.txt")
            with open(WEIGHTS_PATH, 'w') as weight_f:
                 weight_f.write(output)
            # #print output




            # Usage: evaluate.py weights.txt
            # test_data.txt test_labels.txt folder_with_mapper
            command_line = [EVALUATE_SCRIPT]
            command_line.append(WEIGHTS_PATH)
            command_line.append(os.path.join(WORKING_DIR, "test_data.txt"))
            command_line.append(os.path.join(WORKING_DIR, "test_labels.txt"))
            command_line.append(WORKING_DIR)

            #command_line = " ".join(command_line)


            evaluate.main(command_line)


