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
from scipy.ndimage.interpolation import shift
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout


lowval=0

## Stochastic part

# various kinds of networks to be used all using Keras
# DNN
def train_dnn(X_train, Y_train):
	indim=np.shape(X_train)[1]
	outdim=np.shape(Y_train)[1]
	hidden_neurons=400
	model=Sequential()
	model.add(Dense(hidden_neurons,input_dim=indim))
	model.add(Activation("sigmoid"))
	model.add(Dropout(0.5))
	model.add(Dense(hidden_neurons))
	model.add(Activation("sigmoid"))
	model.add(Dropout(0.5))
	model.add(Dense(hidden_neurons))
	model.add(Activation("sigmoid"))
	model.add(Dropout(0.5))
	model.add(Dense(output_dim=outdim))
	model.add(Activation("linear"))
	model.compile(loss="mean_squared_error",optimizer="adam")
	model.fit(X_train,Y_train,batch_size=100, nb_epoch=10, validation_split=0.05)
	return model

def test_dnn(X_test, model):
	return model.predict(X_test)
# RNN

## Deterministic part

#Convolutional nmf reconstruction
def reconstructV(W,H):
	ndim=np.shape(W)[0]
	rdim=np.shape(W)[1]
	twin=np.shape(W)[2]
	ncols=np.shape(H)[1]
	V=np.zeros((ndim,ncols),float)
	for i in range(twin):
		wi=np.squeeze(W[:,:,i])
		hi=shift(H,i)
		V+=np.dot(wi,hi)
	return V

#Convolutional nmf reconstruction from streams
def reconstructV_dsnmf(Wstream,Hstream):
	rdim=np.shape(Hstream)[0]
	ncols=np.shape(Hstream)[1]
	twin=np.shape(np.squeeze(Wstream[0]))[2]
	ndim=np.shape(np.squeeze(Wstream[0]))[0]
	V=np.zeros((ndim,twin-1),float)
	for i in range(twin,ncols+1):
		witr=np.squeeze(Wstream[i-1])
		htmp=Hstream[:,np.arange(i-twin,i)]
		vtmp=reconstructV(witr,htmp)
		vtmp=np.transpose(np.asmatrix(vtmp[:,twin-1]))
		V=np.hstack((V,vtmp))
	return V

#Update H given W
def updateH(V,Vhat,W,H):
	ndim=np.shape(W)[0]
	rdim=np.shape(W)[1]
	twin=np.shape(W)[2]
	ncols=np.shape(H)[1]
	scale=np.divide(V,Vhat)
	onm=np.ones((ndim,ncols))
	newH=np.zeros((rdim,ncols))
	for i in range(twin):
		wi=np.squeeze(W[:,:,i])
		vshift=shift(scale,-1*i)
		numerator=np.dot(np.transpose(wi),vshift)
		denominator=np.dot(np.transpose(wi),onm)
		denominator[denominator < lowval]=lowval
		tmph=np.divide(numerator,denominator)
		tmph[np.isnan(tmph)]=lowval
		newH+=np.multiply(tmph,H)
	return np.divide(newH,twin)

#Update P(W|H)
#def updateWgivenH(V,W,H):

def smaragdis_updates(V,W,H):
	Vhat=reconstructV(W,H)
	print(np.linalg.norm(np.subtract(V,Vhat)))
	Wnew=UpdateW(V,Vhat,W,H)
	W=Wnew
	Vhat=reconstructV(W,H)
	Hnew=updateH(V,Vhat,W,H)
	H=preprocessing.normalize(Hnew,norm='l1')
	return Vhat, W, H

#Update W for a given H
def UpdateW(V, Vhat, W, H):
	ndim=np.shape(W)[0]
	rdim=np.shape(W)[1]
	twin=np.shape(W)[2]
	ncols=np.shape(H)[1]
	scale=np.divide(V,Vhat)
	onm=np.ones((ndim,ncols))
	newW=np.empty((ndim,rdim,twin),float)
	for i in range(twin):
		wi=np.squeeze(W[:,:,i])
		#hi=np.roll(H,i)
		hi=shift(H,i)
		numerator=np.dot(scale,np.transpose(hi))
		denominator=np.dot(onm,np.transpose(hi))
		denominator[denominator < lowval]=lowval
		wtmp=np.divide(numerator,denominator)
		wtmp[np.isnan(wtmp)]=lowval
		wtmpi=np.multiply(wi,wtmp)
		newW[:,:,i]=wtmpi
	return newW
		
#UpdateW stream for dsnmf
def updateWstream(V, Vhat, Wstream, Hstream):
	ndim=np.shape(V)[0]
	rdim=np.shape(H)[0]
	twin=np.shape(np.squeeze(Wstream[0]))[2]
	ncols=np.shape(Hstream)[1]
	newWstream=[]
	for i in range(ncols-twin+1):
		wi=np.squeeze(Wstream[i])
		hi=Hstream[:,i:twin+i]
		vi=V[:,i:i+twin]
		vhat=Vhat[:,i:i+twin]
		newW=UpdateW(vi,vhat,wi,hi)
		newWstream.append(newW)
	return newWstream

#UpdateH stream for dsnmf
def UpdateHstream(v,wstream,hstream):
	ndim=np.shape(v)[0]
	rdim=np.shape(hstream)[0]
	twin=np.shape(np.squeeze(wstream[0]))[2]
	ncols=np.shape(hstream)[1]
	newHstream=np.zeros((rdim,ncols),float)
	newHcnt=np.zeros((rdim,ncols),int)
	score=0
	for i in range(ncols-twin-1):
		wi=np.squeeze(wstream[i])
		hi=hstream[:,i:twin+i]
		vi=v[:,i:i+twin]
		for itr in range(2):
			vhattmp=reconstructV(wi,hi)
			hi=updateH(vi,vhattmp,wi,hi)
			score=np.linalg.norm(np.subtract(vi,vhattmp))
		newH=hi
		newHstream[:,i]=newH[:,0]
	return newHstream

def dsnmf_updates(alldata,allpred,alldatawstream,alldatahstream,ndim,rdim,twin):
	nfiles=np.shape(alldata)[0]
	allX=np.empty((0,((2*twin)+1)*rdim),float)
	allY=np.empty((0,ndim,rdim,twin),float)
	models=[]
	rdim=np.shape(alldatahstream[0])[0]
	for i in range(nfiles):
		wstream=np.squeeze(alldatawstream[i])
		hstream=np.squeeze(alldatahstream[i])
		v=np.squeeze(alldata[i])
		vhat=np.squeeze(allpred[i])
		v=np.transpose(v)
		updateWstream(v,vhat,wstream,hstream)
		tmpa,tmpb=make_traindata(wstream,hstream,twin)
		print(np.shape(tmpa),np.shape(tmpb))
		print(np.shape(tmpa),np.shape(tmpb))
		#allX.append(tmpa)
		#allY.append(tmpb)
		allX=np.vstack((allX,tmpa))
		allY=np.vstack((allY,tmpa))
	#Train P(W|H) over all training data
	nx=np.shape(allX)[0]
	print(np.shape(allX))
	print(np.shape(allY))
	#ndim=np.shape(tmpb)[1]
	#rdim=np.shape(tmpb)[3]
	#twin=np.shape(tmpb)[4]
	for i in range(rdim):
		tmpallY=allY[:,:,i,:].reshape(nx,ndim*twin)
		tmpmodel=train_dnn(allX,tmpallY)
		models.append(tmpmodel)
	#Predict Wstream over all sentences
	for i in range(nfiles):
		hstream=np.squeeze(alldatahstream[i])
		tmpa=make_testdata(hstream,twin)
		tmpb=predictws(tmpa,models,ndim,rdim,twin)
		alldatawstream[i]=tmpb
	#ReconstructV for all sentences
	allpred=[]
	for i in range(nfiles):
		hstream=np.squeeze(alldatahstream[i])
		wstream=alldatawstream(i)
		predv=reconstructV_dsnmf(wstream,hstream)
		allpred.append(predv)
	#UpdateH for all sentences
	for i in range(nfiles):
		hstream=np.squeeze(alldatahstream[i])
		wstream=np.squeeze(alldatawstream[i])
		v=np.squeeze(alldata[i])
		v=np.transpose(v)
		vhat=np.squeeze(allpred[i])
		alldatahstream[i]=UpdateHstream(v,wstream,hstream)
	#ReconstructV for all sentences allpred
	allpred=[]
	score=0
	for i in range(nfiles):
		hstream=np.squeeze(alldatahstream[i,:,:])
		wstream=alldatawstream[i]
		v=alldata[i]
		predv=reconstructV_dsnmf(wstream,hstream)
		score+=np.linalg.norm(np.subtract(v,vhat))
		allpred.append(predv)
	return allpred,alldatawstream, alldatahstream, models, score

## Misc functions
def predictws(xdata,models,ndim,rdim,twin):
	nf=np.shape(xdata)[0]
	wstream=np.empty((nf,ndim,rdim,twin))
	for i in range(rdim):
		tmpmodel=models(i)
		tmpw=test_dnn(xdata,tmpmodel)
		tmpw.reshape(nf,ndim,twin)
		wstream[:,:,i,:]=tmpw
	return wstream

def load_files(ctlfile):
	alldata = []
	fp=open(ctlfile,"r")
	for emaline in fp:
		ln = emaline.strip()
		tmp = np.loadtxt(ln)
		alldata.append(np.asarray(tmp))
	return alldata


def make_wstream_init(H,W):
	wstream=[]
	ncols=np.shape(H)[1]
	for i in range(ncols):
		wstream.append(W)
	return wstream

def make_data_smaragdis(nfiles,alldata,twin,ndim,scaler,halfwin,remwin,minn):
	normV=np.empty((0,ndim),float)
	V=np.empty((0,ndim),float)
	tandemV=np.empty((0,twin*ndim),float)

	## tandem tensor creation for smaragdis updates
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
	return V,normV,tandemV

def make_traindata(wstream,hstream,padwin):
	ncols=np.shape(hstream)[1]
	Xtr=[]
	Ytr=[]
	for i in range(padwin,ncols-padwin):
		xdat=np.ravel(np.squeeze(hstream[:,i-padwin:i+padwin]))
		ydat=np.squeeze(wstream[i,:,:,:])
		#ydat=np.flatten(np.squeeze(wstream[i,:,:,:]))
		Xtr.append(np.asarray(xdat))
		Ytr.append(ydat)

	return Xtr, Ytr

def make_testdata(hstream,padwin):
	ncols=np.shape(hstream)[1]
	rdim=np.shape(hstream)[0]
	Xtr=np.zeros((padwin,rdim*((2*padwin)-1)))
	for i in range(padwin,ncols-padwin):
		xdat=np.flatten(hstream[:,i-padwin:i+padwin])
		#ydat=np.flatten(np.squeeze(wstream[i,:,:,:]))
		Xtr.append(np.asarray(Xtr))
	Xtr.append(np.zeros((padwin,rdim*((2*padwin)-1))))
	return Xtr()

	
#### Main work #######

### parse input arguments
ctlfile = sys.argv[1]
rdim = int(sys.argv[2])
twin = int(sys.argv[3])
halfwin=int(np.floor(twin/2))
remwin=twin-halfwin

# data load into global variable alldata
alldata=load_files(ctlfile)

# data preprocessing - normalizations, make non-negative etc.
nfiles = np.shape(alldata)[0]
ndim = np.shape(alldata[0])[1]

#make global tensor V
unnormV=np.empty((0,ndim),float)
for i in range(nfiles):
	unnormV=np.vstack((unnormV, alldata[i]))

# initial stats
scaler = preprocessing.StandardScaler().fit(unnormV)
normV = scaler.transform(unnormV)
np.savetxt('unnormV.txt',unnormV)
minn = normV.min()
V, normV, tandemV = make_data_smaragdis(nfiles,alldata,twin,ndim,scaler,halfwin,remwin,minn)
print(np.shape(tandemV))

# k means clustering (mahalanobis ?)
kmeans=KMeans(n_clusters=rdim,n_jobs=7,init='k-means++').fit(tandemV)
print(np.shape(kmeans.cluster_centers_))

# init H as cosine distance between tandem tensors and respective k-mean centroids
nr=np.shape(V)[0]
tmpa=preprocessing.normalize(kmeans.cluster_centers_,norm='l2')
tmpb=preprocessing.normalize(tandemV,norm='l2')
H=np.matmul(tmpa,tmpb.transpose())
H=preprocessing.normalize(H,norm='l1')
W=np.random.rand(ndim,rdim,twin)
V=np.transpose(V)

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
for itr in range(4):
	Vhat,W,H=smaragdis_updates(V,W,H)

score1=0
score2=0
alldatahstream=[]
alldatawstream=[]
allpred=[]
for i in range(nfiles):
	nc=np.shape(np.squeeze(alldata[i]))[0]
	v=np.squeeze(alldata[i])
	v=np.transpose(v)
	hstream=np.random.rand(rdim,nc)
	wstream=make_wstream_init(hstream,W)
	alldatawstream.append(wstream)
#	for itr in range(10):
	vhat=reconstructV_dsnmf(wstream,hstream)
	score1=np.linalg.norm(np.subtract(v,vhat))
	hstream=UpdateHstream(v,wstream,hstream)
	alldatahstream.append(hstream)
	vhat=reconstructV_dsnmf(wstream,hstream)
	allpred.append(vhat)
	score2=np.linalg.norm(np.subtract(v,vhat))
	print("before:",score1,"after:",score2)

for itr in range(200):
	allpred,alldatawstream, alldatahstream, models,score=dsnmf_updates(alldata,allpred,alldatawstream,alldatahstream,ndim,rdim,twin)
	print(score)
# Update P(W|H)
#	- Foreach sentence
#	- Process H as necessary (smoothH or otherwise)
#	- Update WgivenH

# trained activations and network

# Update P(W|H) 
H=np.transpose(H)
V=np.transpose(V)
Vhat=np.transpose(Vhat)
np.savetxt('H.txt',H)
np.savetxt('V.txt',V)
np.savetxt('newV.txt',Vhat)
np.save('W.npy',W)
