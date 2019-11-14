# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 16:40:14 2018

@author: Adam
"""

import matplotlib.pyplot as plt
import SingleLayerNet
import numpy as np
import time
#load data
#plt.ion()
#X,Y = dat.make_circles(n_samples = 200,noise = 0.05)
#X = X.T
#Y = Y.T
X=np.load("xFile.npy")
Y=np.load("yFile.npy")
#set up points
plt.figure(1)
plt.subplot(2,1,1)
c = 0
for i in X.T:
    if(Y[c] == 0):
        plt.scatter(i[0],i[1],c="red")
    else:
        plt.scatter(i[0],i[1],c="blue")
    c=c+1
#train
t = time.time()
param = SingleLayerNet.trainNet(X,Y,5,5000,1)
print("Training time " + str(time.time()-t) + " seconds" )
#graph boundary
step = .02
x_min,x_max = X[0].min(), X[0].max()
y_min,y_max = X[1].min(), X[1].max()
xx,yy = np.meshgrid(np.arange(x_min,x_max,step),np.arange(y_min,y_max,step))
mesh = [xx.ravel(),yy.ravel()]
predict = SingleLayerNet.predict(mesh,param)
predict = predict.reshape(xx.shape)
predict = predict.astype(int)
plt.contour(xx,yy,predict)
plt.subplot(2,1,2)
plt.plot(param["cost"])
plt.draw()