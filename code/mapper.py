#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.
import sys
import numpy as np 
from sklearn.kernel_approximation import RBFSampler


# CONFIG
DIMENSION = 400  # Dimension of the original data.
CLASSES = [+1, -1]   # The classes that we are trying to predict.

BATCHSIZE = 200
D = 1000
LAMBDA = 2

# Transformation
# Set up z(x) always during start up (regardless whether included in prediction or validation)
#np.random.seed(42)
#fourier_gaussian = lambda omega: [((2.*np.pi)**(-float(D)/2.)) * np.exp(-np.linalg.norm(omega)**2/2.)]*DIMENSION
#omegas = map(fourier_gaussian, np.random.rand(D,DIMENSION)*10)
np.random.seed(42)
omegas = np.random.multivariate_normal(np.zeros(DIMENSION-100),300*np.eye(DIMENSION-100),D)
bs = 2. *np.pi*np.random.random(D)
prefactor = np.sqrt(2) # precalculate for performance issues
#z = lambda x: prefactor * np.array(map( lambda omega,b: np.cos(np.dot(omega,x)+b) ,omegas,bs))
LEAST_IMPORTANT_DIMENSIONS = [ 5, 12, 20, 25, 26, 33, 36, 37, 38, 40, 49, 50, 67, 68, 74, 75, 84, 87, 89, 91, 94, 96, 97, 99, 101, 105, 106, 107, 110, 117, 
                      123, 125, 137, 139, 148, 152, 153, 154, 166, 167, 168, 169, 173, 177, 188, 193, 199, 201, 202, 203, 206, 214, 217, 220, 223,
                      225, 235, 236, 242, 244, 245, 248, 250, 253, 254, 256, 257, 258, 272, 280, 284, 285, 287, 295, 297, 300, 302, 308, 313, 316,
                      325, 326, 330, 337, 338, 339, 353, 355, 356, 360, 367, 372, 380, 382, 385, 386, 387, 390, 395, 399]

# The call for the transformation
rbf_feature = RBFSampler(gamma=5, n_components=800, random_state=42)
def transform(x_original):
    #return x_original
    #return rbf_feature.fit_transform(x_original)[0]
    #x_transformed = x_original
    #x_transformed = np.concatenate((x_original,[1]),axis=0)
    #return prefactor * np.array(map( lambda omega,b: np.cos(np.dot(omega,x_original)+b) ,omegas,bs))
    return np.delete(x_original, LEAST_IMPORTANT_DIMENSIONS)
    #return prefactor * np.array(map( lambda omega,b: np.cos(np.dot(omega,x_transformed)+b) ,omegas,bs))



def main(stream):
    # read and format data
    w = np.zeros([D]) # weight vector with 0
    S_Adagrad = np.ones([D])  # adagrad scaler with 1
    currentBatch = ""
    for i, line in enumerate(stream):
        if i % BATCHSIZE == 0 and i != 0:

            # Prepare current batch
            A = np.fromstring(currentBatch, sep=' ')
            A = np.reshape(A,(-1,DIMENSION+1))
            Y = A[:, 0]
            X = A[:, 1:DIMENSION+1]
            currentBatch = ""


            # Transformation for each x
            x_transformed = np.array([transform(x) for x in X])

            # PEGASOS for the current Batch
            # create A+ of subgrads (immediatly the sum)
            A_plus = 0

            for j in xrange(x_transformed.shape[0]):
                if Y[j]*np.dot(w,x_transformed[j,:]) < 1:
                    A_plus += Y[j]*x_transformed[j,:]
            # create the gradient
            subgrad = LAMBDA * w - 1./BATCHSIZE*A_plus

            # adapt adagrad parameter S
            S_Adagrad = S_Adagrad + subgrad**2

            # current learning rate
            ETA_t = 1/((float(i)/float(BATCHSIZE)) * LAMBDA)

            # calculate updated w
            w = w - ETA_t / np.sqrt(S_Adagrad) * subgrad
            w = np.min([1, 1/np.sqrt(LAMBDA)/np.linalg.norm(w)])*w


        currentBatch += line

    # format output
    print ' '.join(str(wi) for wi in np.nditer(w))
    #print result


if __name__ == "__main__":
    main(sys.stdin)




# DEPRECATED

# Setup a transformer globally
# TRANSFORMER = Nystroem(gamma=.2, random_state=1) #RBFSampler(gamma=.2, random_state=1)
# Y_TRANSFORMER_FIT = 
# X_TRANSFORMER_FIT = 
# TRANSFORMER.fit(X_TRANSFORMER_FIT, Y_TRANSFORMER_FIT)

#(only fit once in the beginning)
            #if firstBatch:
            #	np.set_printoptions(threshold=sys.maxint)
            #	xFile = open("X.csv", "w")
            #	yFile = open("Y.csv", "w")
            #	xFile.write(repr(X))
            #	yFile.write(repr(Y))