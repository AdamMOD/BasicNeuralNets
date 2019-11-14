# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 15:52:58 2018

@author: Adam
"""
import numpy as np
def sigmoid(x):
  return 1/(1+np.exp(-x))
def initialize(n_x,n_h,n_y):
    #initializeee
    W1 = np.random.randn(n_h,n_x) *.01
    B1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h) *.01
    B2 = np.zeros((n_y,1))
    #save
    parameters = {"W1":W1,
                  "W2":W2,
                  "B1":B1,
                  "B2":B2,
                  "W1hist" : []}
    return parameters

def forward_prop(X,parameters):
    #load
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    B1 = parameters["B1"]
    B2 = parameters["B2"]
    #calculate forward prop vals
    Z1 = np.dot(W1,X)+B1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1)+B2
    A2 = sigmoid(Z2)
    #save
    parameters["Z1"] = Z1
    parameters["Z2"] = Z2
    parameters["A1"] = A1
    parameters["A2"] = A2
    return parameters

def cost_calc(Y,parameters):
    #load
    A2 = parameters["A2"]
    m = Y.shape[0]
    #calculate cost for current model
    inside = np.log(A2)*Y + (1-Y)*(np.log(1-A2))
    summation = (-1/m)*np.sum(inside)
    return summation

def gradient(X,Y,parameters):
    #load
    W2 = parameters["W2"]
    A1 = parameters["A1"]
    A2 = parameters["A2"]
    W2 = parameters["W2"]
    m = Y.shape[0]
    #calculate derivatives 
    dZ2 = A2-Y
    dW2 = (1/m)*np.dot(dZ2,A1.T)
    dB2 = (1/m)*np.sum(dZ2,axis=1,keepdims=True)
    dZ1 = np.dot(W2.T,dZ2)*(1-np.power(A1,2))
    dW1 = (1/m)*np.dot(dZ1,X.T)
    dB1 = (1/m)*np.sum(dZ1,axis=1,keepdims=True)
    #save
    grad = {"dZ2":dZ2,
            "dZ1":dZ1,
            "dW2":dW2,
            "dW1":dW1,
            "dB2":dB2,
            "dB1":dB1}
    return grad
def update(parameters,grads,alpha):
    #load
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    B1 = parameters["B1"]
    B2 = parameters["B2"]
    dW2 = grads["dW2"]
    dW1 = grads["dW1"]
    dB2 = grads["dB2"]
    dB1 = grads["dB1"]
    #change params
    W1 = W1 - alpha*dW1
    W2 = W2 - alpha*dW2
    B1 = B1 - alpha*dB1
    B2 = B2 - alpha*dB2
    #save
    parameters["W1"] = W1
    parameters["W2"] = W2
    parameters["B1"] = B1
    parameters["B2"] = B2
    parameters["W1hist"].append(W1)
    return parameters
    
def trainNet(X,Y,n_h,iterations,alpha):
    n_x = X.shape[0]
    n_y = 1
    cost = []
    initParameters = initialize(n_x,n_h,n_y)
    for i in range(0,iterations):
        params = forward_prop(X,initParameters)
        grads = gradient(X,Y,params)
        params = update(params,grads,alpha)
        if(i%100==0):
            cost.append(cost_calc(Y,params))
            #print(params["W1"],params["W2"],params["B1"],params["B2"])
    params["cost"] = cost
    return params

def predict(X,parameters):
    p = forward_prop(X,parameters)
    predictions = p["A2"] > .5
    return predictions
        
        
    
    