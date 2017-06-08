## AUM
## An algorithm to perform decomposition of timeseries data into deterministic and
## stochastic components, the latter being conditionally dependent on the former.
## Combines ideas of non-negative matrix factorization (for the determinstic part) 
## and deep learning (for the stochastic part).

## environment and imports
import sys
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans

## Stochastic part

# various kinds of networks to be used all using Keras
# DNN
# RNN

## Deterministic part

# convolutional nmf reconstruction

# Update H given W

# Update P(W|H) 

## Misc functions

#### Main work #######

# parse input arguments
ctlfile = sys.argv[1]
rdim = int(sys.argv[2])
twin = int(sys.argv[3])
halfwin=int(np.floor(twin/2))
remwin=twin-halfwin

# data load into global variable alldata
alldata = []
fp=open(ctlfile,"r")
for emaline in fp:
	ln = emaline.strip()
	tmp = np.loadtxt(ln)
	alldata.append(np.asarray(tmp))
	
# data preprocessing - normalizations, make non-negative etc.
nfiles = np.shape(alldata)[0]
ndim = np.shape(alldata[0])[1]
print nfiles,ndim
#make global tensor V
unnormV=np.empty((0,ndim),float)
for i in range(nfiles):
	unnormV=np.vstack((unnormV, alldata[i]))

# initial stats
scaler = preprocessing.StandardScaler().fit(unnormV)
normV = scaler.transform(unnormV)
minn = normV.min()
normV=np.empty((0,ndim),float)
V=np.empty((0,ndim),float)
tandemV=np.empty((0,twin*ndim),float)

## tandem tensor creation
for i in range(nfiles):
	tempV=alldata[i]
	nr=np.shape(tempV)[0]
	normtempV=scaler.transform(tempV)
	nonnegtempV=normtempV[halfwin:nr-remwin,:]-minn
	normV=normtempV-minn
	V=np.vstack((V,nonnegtempV))
	tmptandv=np.empty((nr-twin,0),float)
	for j in range(-1*halfwin,remwin):
		tmpb=normV[halfwin+j:nr-remwin+j,:]
		tmptandv=np.hstack((tmptandv,tmpb))

	tandemV=np.vstack((tandemV,tmptandv))

	
print np.shape(V)
print np.shape(tandemV)

# k means clustering (mahalanobis ?)
kmeans=KMeans(n_clusters=rdim,n_jobs=7).fit(tandemV)
print np.shape(kmeans.cluster_centers_)

# init H as cosine distance between tandem tensors and respective k-mean centroids

# loop windows of size \tau through dataframe and compute instantaneous Ws 
# via spikyH and WgivenH
# or Ws and Hs, ignore the Hs over each analysis window
# or process sets of trails in batch, ignore Hs

# Train a network for predicting W given H

## Unto convergence by looping over all data

# Update H given E[W|H]
#	- Foreach sentence
# 	- Estimate W tensor from trained network
#	- Update HgivenW over several iterations

# Update P(W|H)
#	- Foreach sentence
#	- Process H as necessary (smoothH or otherwise)
#	- Update WgivenH

# trained activations and network

# Update P(W|H) 
