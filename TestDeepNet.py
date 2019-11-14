# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 16:40:14 2018

@author: Adam
"""

import matplotlib.pyplot as plt
import DeepNet as dNet
import numpy as np
import time
#load data
X=np.load("xFile.npy")
Y=np.load("yFile.npy")
#set up points
plt.figure(2)
plt.subplot(2,1,1)
c = 0
for i in X.T:
    if(Y[c] == 0):
        plt.scatter(i[0],i[1],c="red")
    else:
        plt.scatter(i[0],i[1],c="blue")
    c=c+1
#train
dims = [2,3,1]
t = time.time()
param,grad = dNet.trainNet(X,Y,dims,5000,.1)
print("Training time " + str(time.time()-t) + " seconds" )
print(int(len(param)/2))
#graph boundary
step = .02
x_min,x_max = X[0].min(), X[0].max()
y_min,y_max = X[1].min(), X[1].max()
xx,yy = np.meshgrid(np.arange(x_min,x_max,step),np.arange(y_min,y_max,step))
mesh = [xx.ravel(),yy.ravel()]
predict = dNet.predict(mesh,param)
predict = predict.reshape(xx.shape)
predict = predict.astype(int)
plt.contour(xx,yy,predict)
plt.subplot(2,1,2)
plt.plot(param["cost"])
plt.draw()
#write
f = open("Grads3Layer.txt",'w')
for i in range(1,len(dims)):
    print(param['W' + str(i)])
#f.write(str(grad))
f.close()