#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.
import sys
import numpy as np 
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

# CONFIG
DIMENSION = 400  # Dimension of the original data.
CLASSES = [+1, -1]   # The classes that we are trying to predict.
BATCHSIZE = 100
D = 1000

# The call for the transformation
LEAST_IMPORTANT_DIMENSIONS = [ 5, 12, 20, 25, 26, 33, 36, 37, 38, 40, 49, 50, 67, 68, 74, 75, 84, 87, 89, 91, 94, 96, 97, 99, 101, 105, 106, 107, 110, 117, 
					  123, 125, 137, 139, 148, 152, 153, 154, 166, 167, 168, 169, 173, 177, 188, 193, 199, 201, 202, 203, 206, 214, 217, 220, 223, 
					  225, 235, 236, 242, 244, 245, 248, 250, 253, 254, 256, 257, 258, 272, 280, 284, 285, 287, 295, 297, 300, 302, 308, 313, 316, 
					  325, 326, 330, 337, 338, 339, 353, 355, 356, 360, 367, 372, 380, 382, 385, 386, 387, 390, 395, 399]
VARIANCE = 300

np.random.seed(42)
omegas = np.random.multivariate_normal(np.zeros(DIMENSION),300*np.eye(DIMENSION),D)
bs = 2. *np.pi*np.random.random(D)
prefactor = np.sqrt(2) # precalculate for performance issues

rbf_feature = None
clf = None #LinearSVC(C=1.0, fit_intercept=False,  tol=0.00001)


def transform(x_original):
	return rbf_feature.transform(x_original)[0]
	#x_transformed = rbf_feature.fit_transform(x_original)
	#print x_original.shape, "\t", x_transformed.shape, "\t", x_transformed[0]
	#return x_transformed[0]
	#return prefactor * np.array(map( lambda omega,b: np.cos(np.dot(omega,x_original)+b) ,omegas,bs))
	#return np.delete(x_original, LEAST_IMPORTANT_DIMENSIONS)
	#return prefactor * np.array(map( lambda omega,b: np.cos(np.dot(omega,x_original)+b) ,omegas,bs))



def main(stream):
	data = np.fromstring(stream, sep=' ')
	data = np.reshape(data,(-1,DIMENSION+1))
	Y = data[:, 0]
	X = data[:, 1:DIMENSION+1]

	
	# Transform
	x_transformed = np.array([transform(x) for x in X])
	

	# Set up classifier and pipeline
	# parameters = {
	#        	'classifier__C': [0.1, 1, 10],
	#       	'classifier__gamma': [1],
	#       	"classifier__tol": [0.001, 0.000001],
	#       	'classifier__cache_size': 500
	#    }
	#parameters = {
	#	'estimators ': [10, 40, 60],
	#	'criterion ': ["gini", "entropy"],
	#	'n_jobs': [-1]
	#}
	#pipeline = Pipeline([("classifier", RandomForestClassifier())])
	
	# Fit 
	#gs = GridSearchCV(RandomForestClassifier(), parameters, verbose=2, refit=False, cv=6, n_jobs = 1)

	clf.fit(x_transformed, Y)

	# format output
	result = ' '.join(' '.join(str(cell) for cell in row) for row in clf.coef_)
	return result

if __name__ == "__main__":
	currentBatch = sys.stdin.read()
	print main(currentBatch)
