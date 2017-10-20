import numpy
import urllib
import scipy.optimize
import random
from math import exp
from math import log
from urllib import request

def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)

print ("Reading data...")
data = list(parseData("http://jmcauley.ucsd.edu/cse255/data/beer/beer_50000.json"))
print ("done")

def inner(x,y):
  return sum([x[i]*y[i] for i in range(len(x))])

def sigmoid(x):
  return 1.0 / (1 + exp(-x))

# NEGATIVE Log-likelihood
def f(theta, X, y, lam):
  loglikelihood = 0
  for i in range(len(X)):
    logit = inner(X[i], theta)
    loglikelihood -= log(1 + exp(-logit))
    if not y[i]:
      loglikelihood -= logit
  for k in range(len(theta)):
    loglikelihood -= lam * theta[k]*theta[k]
  #print ("ll =", loglikelihood)
  return -loglikelihood

# NEGATIVE Derivative of log-likelihood
def fprime(theta, X, y, lam):
  dl = [0.0]*len(theta)
  for k in range(len(theta)):
      res = 0
      for i in range(len(X)):
          logit = inner(X[i], theta)
          res += X[i][k]*(1-sigmoid(logit))
          if not y[i]:
              res -=X[i][k]
      res -= 2*lam*theta[k]
      dl[k] = res
  # Negate the return value since we're doing gradient *ascent*
  return numpy.array([-x for x in dl])


X = [[b['beer/ABV'],b['review/taste']] for b in data]
y = ["American IPA" in b['beer/style'] for b in data]

X_train = X[:int(len(X)/2)]
y_train = y[:int(len(X)/2)]
X_test = X[int(len(X)/2):]
y_test = y[int(len(X)/2):]


# If we wanted to split with a validation set:
#X_valid = X[len(X)/2:3*len(X)/4]
#X_test = X[3*len(X)/4:]

# Use a library function to run gradient descent (or you can implement yourself!)
theta,l,info = scipy.optimize.fmin_l_bfgs_b(f, [0]*len(X[0]), fprime, args = (X_train, y_train, 1.0))
print ("Final log likelihood =", -l)
print (theta)

y_predicted = [0] * len(X_test)
for i in range(len(X_test)):
    if inner(theta, X_test[i]) > 0:
        y_predicted[i] = 1

match_test  = [(x==y) for x,y in zip(y_predicted, y_test)]
print ("Accuracy = ", sum(match_test)*1.0/len(match_test)) # Compute the accuracy
