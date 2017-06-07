## AUM
## An algorithm to perform decomposition of timeseries data into deterministic and
## stochastic components, the latter being conditionally dependent on the former.
## Combines ideas of non-negative matrix factorization (for the determinstic part) 
## and deep learning (for the stochastic part).

## environment and imports
import sys
import numpy as np
import pandas as pd

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
ctlfile=sys.argv[1]

# data load into global variable alldata
alldata = []
fp=open(ctlfile,"r")
for emaline in fp:
	ln = emaline.strip()
	tmp = np.loadtxt(ln)
	alldata.append(np.asarray(tmp))

# data preprocessing - normalizations, make non-negative etc.

# initial work

## tandem tensor creation

# k means clustering (mahalanobis)

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
