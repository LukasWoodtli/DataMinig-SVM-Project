#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.
import sys
import numpy as np 
import time
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

# CONFIG
DIMENSION = 400  # Dimension of the original data.
CLASSES = [+1, -1]   # The classes that we are trying to predict.
D = 1000

stdScaler = StandardScaler()

#precomputed attributes of the data
LEAST_IMPORTANT_DIMENSIONS = [ 5, 12, 20, 25, 26, 33, 36, 37, 38, 40, 49, 50, 67, 68, 74, 75, 84, 87, 89, 91, 94, 96, 97, 99, 101, 105, 106, 107, 110, 117, 
					  123, 125, 137, 139, 148, 152, 153, 154, 166, 167, 168, 169, 173, 177, 188, 193, 199, 201, 202, 203, 206, 214, 217, 220, 223,
					  225, 235, 236, 242, 244, 245, 248, 250, 253, 254, 256, 257, 258, 272, 280, 284, 285, 287, 295, 297, 300, 302, 308, 313, 316,
					  325, 326, 330, 337, 338, 339, 353, 355, 356, 360, 367, 372, 380, 382, 385, 386, 387, 390, 395, 399]

MEANS = np.array([ 0.00211025,  0.00324696,  0.00337556,  0.00174057,  0.00354896,
		0.00365411,  0.00191883,  0.00217688,  0.00189749,  0.00345257,
		0.00522937,  0.00450376,  0.00357557,  0.00303662,  0.0022213 ,
		0.00177112,  0.00217466,  0.00383929,  0.0025618 ,  0.00288916,
		0.00610232,  0.00237953,  0.00435742,  0.00716557,  0.00365985,
		0.000575  ,  0.00308115,  0.00624082,  0.00184208,  0.00182251,
		0.00328006,  0.00346353,  0.00244753,  0.00261274,  0.00362024,
		0.00306702,  0.00213377,  0.00122881,  0.00210671,  0.00305775,
		0.00161729,  0.01251365,  0.00251241,  0.00525508,  0.00266099,
		0.00282528,  0.00389699,  0.00208015,  0.00309528,  0.00202472,
		0.00407467,  0.00183953,  0.00315654,  0.00350098,  0.00413939,
		0.00346747,  0.00069988,  0.00126979,  0.00235875,  0.00392572,
		0.00241762,  0.0121345 ,  0.00127452,  0.002694  ,  0.00432321,
		0.0006373 ,  0.00221361,  0.00080185,  0.00277273,  0.00118416,
		0.00336503,  0.00341899,  0.00262726,  0.00096613,  0.00217728,
		0.00261086,  0.00151708,  0.00303089,  0.004046  ,  0.00276146,
		0.00079088,  0.00262003,  0.00269931,  0.00279556,  0.00188092,
		0.00240583,  0.00322351,  0.00289572,  0.00125377,  0.00220346,
		0.00214879,  0.00226922,  0.00368343,  0.00284917,  0.00057063,
		0.00311928,  0.00282178,  0.00130878,  0.00072395,  0.00360818,
		0.00329504,  0.00232072,  0.00318968,  0.00386877,  0.00271196,
		0.00098039,  0.00231967,  0.00135151,  0.00466628,  0.00383954,
		0.00043084,  0.00151822,  0.00247882,  0.00181005,  0.00283078,
		0.00055562,  0.00414502,  0.00226014,  0.0036375 ,  0.00194639,
		0.00221299,  0.00210288,  0.00139533,  0.00597269,  0.00210934,
		0.00228058,  0.00119831,  0.00296831,  0.00230649,  0.00087266,
		0.00355678,  0.00229393,  0.00378303,  0.00269556,  0.00197321,
		0.00207433,  0.00145592,  0.00017641,  0.00225186,  0.00353521,
		0.0030318 ,  0.00240654,  0.00042535,  0.00300716,  0.00371683,
		0.0024242 ,  0.0008588 ,  0.00110441,  0.00277935,  0.00136685,
		0.00428111,  0.00325421,  0.00256363,  0.00235525,  0.00212555,
		0.00290243,  0.00169826,  0.00283853,  0.00303707,  0.00320161,
		0.00419415,  0.00052745,  0.00300946,  0.00027639,  0.00262298,
		0.0019836 ,  0.00263868,  0.0001719 ,  0.00231489,  0.00191246,
		0.00351358,  0.00303667,  0.0005869 ,  0.00111067,  0.00364059,
		0.00095534,  0.0018637 ,  0.00286058,  0.00402401,  0.00041539,
		0.00089393,  0.00275739,  0.00211345,  0.00323125,  0.00129552,
		0.00113703,  0.00247091,  0.00240762,  0.00358012,  0.00317464,
		0.00532699,  0.00315006,  0.00302053,  0.00026278,  0.00230282,
		0.00228911,  0.00411986,  0.00237018,  0.00486294,  0.00053128,
		0.00092962,  0.00036782,  0.00079751,  0.00074485,  0.0028313 ,
		0.00362407,  0.00571766,  0.00320049,  0.00140197,  0.00409253,
		0.00270594,  0.00227434,  0.00130327,  0.00306598,  0.00147881,
		0.00190314,  0.00240991,  0.00257926,  0.00189383,  0.0022505 ,
		0.00418135,  0.00237159,  0.00226436,  0.00293675,  0.00079367,
		0.00015837,  0.00068488,  0.0022262 ,  0.00235127,  0.00034002,
		0.00301371,  0.00414148,  0.0018109 ,  0.00460725,  0.00254723,
		0.00218138,  0.00309942,  0.00136598,  0.00110758,  0.0035363 ,
		0.0025288 ,  0.00429634,  0.00091122,  0.00153081,  0.00042751,
		0.00204943,  0.00294187,  0.00057175,  0.00060359,  0.00551554,
		0.00059049,  0.00239804,  0.0051181 ,  0.00323403,  0.00166517,
		0.00243574,  0.00334279,  0.00177908,  0.00035481,  0.00230916,
		0.00424659,  0.00332952,  0.00414395,  0.00091478,  0.00278167,
		0.00208381,  0.00224656,  0.00214092,  0.00339209,  0.00255123,
		0.00420709,  0.00238361,  0.00130355,  0.00383209,  0.0021548 ,
		0.00324393,  0.0025279 ,  0.0039593 ,  0.0032275 ,  0.00356508,
		0.00228288,  0.00208828,  0.00127472,  0.0029878 ,  0.00304852,
		0.00316507,  0.00070435,  0.00230542,  0.0033167 ,  0.00281885,
		0.00339695,  0.00214468,  0.00136332,  0.00253422,  0.00045356,
		0.00141041,  0.00178019,  0.00041719,  0.00072298,  0.00303746,
		0.00140748,  0.00251543,  0.00026139,  0.0044071 ,  0.00453756,
		0.00468382,  0.00101928,  0.0027445 ,  0.00285803,  0.00249545,
		0.00179221,  0.00279827,  0.00080698,  0.00141419,  0.00185972,
		0.00182306,  0.00272871,  0.00317053,  0.00191721,  0.00175408,
		0.00290738,  0.00051098,  0.00419882,  0.0013124 ,  0.00252249,
		0.00316026,  0.0015901 ,  0.00333585,  0.00320708,  0.0016289 ,
		0.0003912 ,  0.00159977,  0.00140486,  0.00222881,  0.00294017,
		0.00286368,  0.00124911,  0.00280589,  0.00016481,  0.00145117,
		0.00259894,  0.00365369,  0.00281812,  0.00319838,  0.00093047,
		0.00229674,  0.00309688,  0.00885282,  0.00197076,  0.00161581,
		0.0030922 ,  0.00174751,  0.00252885,  0.0013867 ,  0.00050075,
		0.00241674,  0.00165337,  0.00139488,  0.00261064,  0.00155389,
		0.00196605,  0.00443576,  0.00248707,  0.00247497,  0.00324396,
		0.00083328,  0.0004641 ,  0.00262795,  0.00301221,  0.00334871,
		0.00306921,  0.00323425,  0.0025591 ,  0.00739358,  0.00263813,
		0.00102726,  0.00186119,  0.00322866,  0.00254241,  0.00107487,
		0.00164742,  0.00225573,  0.00063662,  0.0026686 ,  0.00114404,
		0.00209311,  0.00062432,  0.00127358,  0.00304692,  0.00132683,
		0.00172449,  0.00279628,  0.00470283,  0.00152954,  0.00267596,
		0.00211082,  0.00320547,  0.00028523,  0.00114486,  0.00317149])
VARIANCE = 200

# The call for the transformation
np.random.seed(42)
omegas = np.random.multivariate_normal(np.zeros(DIMENSION-100),VARIANCE*np.eye(DIMENSION-100),D)
bs = 2. *np.pi*np.random.random(D)
prefactor = np.sqrt(2./1000.) # precalculate for performance issues
def transform(x_original):
	#return x_original
	#x_transformed = rbf_feature.fit_transform(x_original)
	#print x_original.shape, "\t", x_transformed.shape, "\t", x_transformed[0]
	#return x_transformed[0]
	x_transformed = x_original
	#x_transformed = x_original - MEANS
	x_transformed = np.delete(np.array(x_transformed), LEAST_IMPORTANT_DIMENSIONS)
	x_transformed = prefactor * np.array(map( lambda omega,b: np.cos(np.dot(omega,x_transformed)+b) ,omegas,bs))
	return stdScaler.fit_transform(x_transformed)




def main(data):

	# prepare data
	data = np.fromstring(data, sep=' ')
	data = np.reshape(data,(-1,DIMENSION+1))
	Y = data[:, 0]
	X = data[:, 1:DIMENSION+1]

	# Do the transformation for each x
	x_transformed = np.array([transform(x) for x in X])


	# fit current batch partially
	# parameters = {
	#     'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'], 'alpha': [1e-5,1e-7,1e-9], 'fit_intercept': [True,False], 'average': [True,10, 100],
	#     'n_iter': [3], 'penalty': ['l1', 'l2', 'elasticnet'], 'eta0': [0, 0.5, 0.8], 'learning_rate': ['optimal', 'constant'], 'epsilon': [0.1, 0.001]
	#     }
	# spaeter noch einen Durchgang mit invscaling und 'power_t': [0.2,0.5,0.8] dann
	#	{
	#	'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'], 'alpha': [0.00000001], 'fit_intercept': [False], 'average': [True],
	#	'n_iter': [5,10], 'power_t': [0.2, 0.5,0.8], 'penalty': ['l1', 'l2'], 'eta0': [0.5], 'learning_rate': ['inv_scaling']
	#	}
	#}

	parameters = {'C': [0.001, 0.5, 2, 100]}

	best_params = {'n_iter': [1], 'warm_start': [False], 'loss': ['hinge'], 'n_jobs': [1], 'eta0': [0.5], 'verbose': [0], 'shuffle': [True], 'fit_intercept': [False], 'epsilon': [0.1], 'average': [True], 'penalty': ['l2'], 'power_t': [0.5], 'random_state': [None], 'l1_ratio': [0.15], 'alpha': [1e-08], 'learning_rate': ['constant'], 'class_weight': [None]}

	abgabe_params = {'loss':['modified_huber'], 'alpha':[0.00000001], 'fit_intercept':[False], 'average':[True], 'n_iter':[8], 'n_jobs':[-1], 'penalty':["l2"]}
	# fit current batch partially
	#clf = SGDClassifier(loss='modified_huber', alpha=0.00000001, fit_intercept=False, average=True, n_iter=10, power_t=0.5, penalty='elasticnet', learning_rate='optimal')
	#score_func = metrics.f1_score
	gs = GridSearchCV(SVC(kernel='linear', probability=True, random_state=23), parameters, verbose=100, refit=True, cv=4, n_jobs = -1)
	gs.fit(x_transformed, Y)


	#{'kernel': 'linear', 'C': 1.0, 'verbose': False, 'degree': 3, 'shrinking': True, 'max_iter': -1, 'random_state': 23, 'tol': 0.001, 'cache_size': 200, 'coef0': 0.0, 'gamma': 0.0, 'class_weight': None}


	# format output
	print gs.best_estimator_.get_params()
	result = ' '.join(' '.join(str(cell) for cell in row) for row in gs.best_estimator_.coef_)
	return result




	#scaler = StandardScaler();
	#scaler.fit(X,Y)
	#print "Scale:\n", scaler.scale_
	#np.set_printoptions(threshold=np.nan)
	#print "Mean:\n", repr(scaler.mean_)


	#scaler = StandardScaler();
	#scaler.fit(X,Y)
	#print "Scale:\n", scaler.scale_
	#np.set_printoptions(threshold=np.nan)
	#print "Mean:\n", repr(scaler.mean_)
	#print "Variance:\n", scaler.var_


	#for i, line in enumerate(sys.stdin):
		# if i % BATCHSIZE == 0 and i != 0:
		#  	# prepare current batch
		# 	data = np.fromstring(currentBatch + lastBatch, sep=' ')
		# 	data = np.reshape(data,(-1,DIMENSION+1))
		# 	Y = data[:, 0]
		# 	X = data[:, 1:DIMENSION+1]

		# 	# Do the transformation for each x
		# 	x_transformed = np.array([transform(x) for x in X])
		# 	#print x_transformed.shape

		# 	# fit current batch partially
		# 	clf.partial_fit(x_transformed, Y, classes=np.array(CLASSES))

		# 	# clear batch
		# 	lastBatch = currentBatch
		# 	currentBatch = ""


		#currentBatch += line

if __name__ == "__main__":

	# read and format data
	data = sys.stdin.read()
	print main(data)