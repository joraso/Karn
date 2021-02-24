# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 13:32:32 2021
@author: Joe Raso
"""

import numpy as np
import tensorflow as tf

class SimpleClassifier:
    """ A neural network design to solve classification problems such as
        image recognition. The basic construction is an input layer, followed
        by any number of interstitial dense layers (specified by 'structure'),
        and an output layer. The output is a normalized vector of size
        ncatagories, which contains something equivalent to the estimated
        likelyhood that the input sample belongs in that catagory.
        
        Arguments:
        inputdims   - (tuple) Dimensions of the input data.
        ncatagories - (int) Number of catagories.
        blocks      - (list) List of DenseBlock building block objects to
                      construct the network from.
                      
        Keywords:
        optimizer   - (str) Keras optimizer to use during training. defaults
                     to 'adam'.
        verbose     - (bool) Toggles command line output. Defaults to True.
        """
    def __init__(self, inputdims, ncatagories, blocks, optimizer='adam',
                 verbose=True):
        self.inputdims = inputdims
        self.ncatagories = ncatagories
        self.blocks = blocks
        self.optimizer = optimizer
        self.verbose = verbose
        # Attributes not created by the user
        self.layers = []; self.model = None
        # History tracking
        self.loss=[]; self.accuracy=[]
        # Now build it.
        self.build()
        
    def build(self):
        """Constructs the model from the give building blocks."""
        # Input layer
        self.layers = [tf.keras.layers.Flatten(input_shape=self.inputdims)]
        # Interstitial layers
        for blk in self.blocks:
            self.layers += blk.layerlist()
        # output layer
        self.layers.append(tf.keras.layers.Dense(self.ncatagories,
                                activation='softmax'))
        # compiling
        self.model = tf.keras.Sequential(self.layers)
        self.model.compile(optimizer=self.optimizer, metrics=['accuracy'],
                           loss='sparse_categorical_crossentropy')
                           
    def train(self, xtrain, ytrain, batch=1, epochs=1):
        """Trains the model on the given data set.
        
            Arguments:
            xtrain - (ndarray) Input training data.
            ytrain - (ndarray) Training labels, expected to be an array of 
                     ints in the range [0, ncatagories).
                     
            Keywords:
            batch  - (int) Batch size. defaults to 1.
            epochs - (int) Number of epochs. defaults to 1.
            """
        self.model.fit(xtrain, ytrain, batch_size=batch, epochs=epochs,
                       verbose=self.verbose, shuffle=True)
        self.loss += (self.model.history.history['loss'])
        self.accuracy += (self.model.history.history['accuracy'])
        
    def predict(self, xset):
        """Returns the catagory predictions on a test data set. Uses the simple
            most likely catagory given the vector output of the model.
            Arguments:
            xset - (ndarray) The set of input data to catagorize. Note that the
                   function expects a batch of samples to predict, so take care
                   when predicting a singular data sample that the dimensions 
                   are consistent with a batch containing many samples (i.e. it
                   should have a 0-axis corrosponding to the sample number).
            """
        pred = self.model.predict(xset)
        return np.argmax(pred, axis=1)
        
        
if __name__ == '__main__':
    
    # Attempted Bias / Variance evaluation:
    (xtrain, ytrain), (xtest, ytest) = tf.keras.datasets.fashion_mnist.load_data()
    from BuildingBlocks import DenseBlock
    block = DenseBlock([784, 196, 196, 49])
    net = ClassifierNet((28,28), 10, [block])
    net.train(xtrain, ytrain, batch=5000, epochs=20)
    ytrue = ytest
    ypred = net.model.predict(xtest)
    ypred = net.predict(xtest)    
    LossFxn = tf.keras.losses.MeanSquaredError() # Perhaps the better loss function
#    LossFxn = tf.keras.losses.SparseCategoricalCrossentropy() # Loss function used in the model
    # find the prediction that has a minimum average loss compared to the others:
    cross_losses = np.zeros(ypred.shape[0])
    for i in range(ypred.shape[0]):
        # Select a test prediction
#        ytest = np.argmax(ypred, axis=1)[i]*np.ones(ypred.shape[0])
        ytest = ypred[i]*np.ones(ypred.shape[0])
        # Record the average loss, with that prediction as the 'true value'
        cross_losses[i] = LossFxn(ytest, ypred).numpy()
    mindx = np.where(cross_losses==np.min(cross_losses))[0][0] # row of the main prediction
    # find the prediction that resulted in the minimum average loss (slice to a single row if needed)
    ymain = ypred[mindx]
    # the bias is the loss value at the main prediction (compared to true)
#    bias = LossFxn(ytrue[mindx], ymain).numpy()
    bias = LossFxn(np.array([ytrue[mindx]]), np.array([ymain])).numpy()
    # finally, the variance is the average loss value between the ymain and ypred
    variance = LossFxn(np.argmax(ypred, axis=1), ymain*np.ones(ypred.shape)).numpy()
    variance = LossFxn(ypred, ymain*np.ones(ypred.shape)).numpy()
