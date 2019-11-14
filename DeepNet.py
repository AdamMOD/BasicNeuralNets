# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 18:06:28 2018

@author: Adam
    Reminder - in params we have all the Ws and Bs, in cache we have all the As and Zs, in grads we have all the dW, dZ, dB, dA
"""
import numpy as np
from copy import deepcopy
hiddenActivation = "sigmoid"
def sigmoid(x):
  return 1/(1+np.exp(-x))
#netDims in form input dimension count, neurons in hidden layer 1, n2, etc, num of output neurons
def initialize(netDims):
    params = {}
    for i in range (1,len(netDims)):
        params["W" + str(i)] = np.random.randn(netDims[i],netDims[i-1]) * .01
        params["B" + str(i)] = np.random.randn(netDims[i],1) * .01
    return params
#propogation for a single layer of the net
def layer_forward_prop(APrev,W,B,activation):
    Z = np.dot(W,APrev)+B
    if(activation == "sigmoid"):
        A = sigmoid(Z)
    elif(activation == "relu"):
        A = np.maximum(0,Z)
    elif(activation == "leakyrelu"):
        A=deepcopy(Z)
        np.copyto(A,.01*A,where=np.less_equal(A,0))
    return Z,A
#propogation for    the net (include X as A0 in cache)
def forward_prop(params,cache):
    L = int(len(params)/2)
    for i in range(1,L):
        cache['Z' + str(i)],cache['A' + str(i)] = layer_forward_prop(cache['A'+str(i-1)],params['W' + str(i)],params['B'+str(i)],hiddenActivation)
    cache['Z' + str(L)],cache['A' + str(L)] = layer_forward_prop(cache['A'+str(L-1)],params['W' + str(L)],params['B'+str(L)],"sigmoid")
    return cache
def cost_calc(Y,cache,L):
    A = cache["A"+ str(L)]
    m = Y.shape[0]
    inside = np.log(A)*Y + (1-Y)*(np.log(1-A))
    summation = (-1/m)*np.sum(inside)
    return summation
def layer_gradient(Z,APrev,W,A,dA,activation):
    m=Z.shape[1]
    if(activation == "relu"):
        dZ = dA*((np.not_equal(0,Z)).astype(int))
        dW = (1/m)*np.dot(dZ,APrev.T)
        dB = (1/m)*np.sum(dZ,axis=1,keepdims=True)
        dAPrev = np.dot(W.T,dZ)
    elif(activation == "leakyrelu"):
        dZ=deepcopy(Z)
        dZ[dZ>0]=1
        dZ[dZ<=0]=.01
        dW = (1/m)*np.dot(dZ,APrev.T)
        dB = (1/m)*np.sum(dZ,axis=1,keepdims=True)
        dAPrev = np.dot(W.T,dZ)
    elif(activation == "sigmoid"):
        dZ = dA*(A*(1-A))
        dW = (1/m)*np.dot(dZ,APrev.T)
        dB = (1/m)*np.sum(dZ,axis=1,keepdims=True)
        dAPrev = np.dot(W.T,dZ)        
    return dW,dB,dAPrev
def gradient(Y,params, cache):
    L = int(len(params)/2)
    AL = cache["A" + str(L)]
    AL[AL==0] = .0001
    AL[AL==1] = .9999
    dAL = (-Y/AL)+((1-Y)/(1-AL))
    grads = {"dA" + str(L) : dAL}
    #last layer first
    grads["dW" + str(L)],grads["dB" + str(L)],grads["dA" + str(L-1)] = layer_gradient(cache["Z" + str(L)],
                                     cache["A" + str(L-1)],params["W" + str(L)],cache["A" + str(L)],dAL,"sigmoid")
    for i in range(L-1,0,-1):
        grads["dW" + str(i)],grads["dB" + str(i)],grads["dA" + str(i-1)] = layer_gradient(cache["Z" + str(i)],
                                     cache["A" + str(i-1)],params["W" + str(i)],cache["A" + str(i)],grads["dA"+str(i)],hiddenActivation)
    return grads
def update(params,grads,alpha):
    L = int(len(params)/2)
    for i in range(1,L+1):
        params["B"+str(i)] = params["B"+str(i)]-alpha*grads["dB"+str(i)]
        params["W"+str(i)] = params["W"+str(i)]-alpha*grads["dW"+str(i)]
    return params
def trainNet(X,Y,netDims,iterations,alpha):
    cost = []
    grados = []
    parameters = initialize(netDims)
    cache = {"A0" : X}
    L = int(len(parameters)/2)
    for i in range(0,iterations):
       cache = forward_prop(parameters,cache)
       gradients = gradient(Y,parameters,cache)
       parameters = update(parameters,gradients,alpha)
       if(i%1==0):
           cost.append(cost_calc(Y,cache,L))
           grados.append(gradients)
    parameters["cost"] = cost
    return parameters,grados
def predict(X,params):
    cache = {"A0" : X}
    L = int(len(params)/2)
    probabilities = forward_prop(params,cache)
    predictions = probabilities["A"+str(L)] > .5
    return predictions