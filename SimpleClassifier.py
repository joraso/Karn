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
        
    def confusion(self, xtest, ytest):
        """Returns a the confusion matrix of the trained model as a numpy array.
        
            Arguments:
            xtrain - (ndarray) Input test data.
            ytrain - (ndarray) Test training labels, expected to be an array of 
                     ints in the range [0, ncatagories).
            
            Returns:
            matrix - (ndarray) The confusion matrix (true values as rows, 
                     predictions as columns).
        """
        matrix = np.zeros((self.ncatagories,self.ncatagories))
        ypred = self.predict(xtest)
        for i in range(len(ypred)):
            matrix[ytest[i], ypred[i]] += 1
        return matrix
        
    def score(self, xtest, ytest, averaged=False):
        """Returns a set of classification error metrics for the model.
        
            Arguments:
            xtrain      - (ndarray) Input test data.
            ytrain      - (ndarray) Test training labels, expected to be an
                          array of ints in the range [0, ncatagories).
                     
            Keywords:
            averaged    - (bool) If True, return the average values of recall,
                          precision and specificity across all catagories in 
                          the model. If False (default), these values are 
                          returned as ndarrays containing the values for each 
                          catagory individually.
                     
            Returns (4-values):
            accuracy    - (float) The accuracy; Tr(confusion matrix)/n_samples.
            recall      - (float or ndarray) Recall; TP/(TP+FN), calculated for
                          each catagory on a one-vs-all bases.
            precision   - (float or ndarray) Precision or sensitivity; 
                          TP/(TP+FN), calculated for each catagory on a
                          one-vs-all bases.
            specificity - (float or ndarray) Specificity; TN/(TN+FP),
                          calculated for each catagory on a one-vs-all bases.
        """        
        matrix = self.confusion(xtest, ytest)
        accuracy = np.trace(matrix)/len(ytest)
        # remaining scores are computed by catagory
        recall = np.zeros(self.ncatagories)
        precision = np.zeros(self.ncatagories)
        specificity = np.zeros(self.ncatagories)
        for i in range(self.ncatagories):
            recall[i] = matrix[i,i] / np.sum(matrix[i,:])
            precision[i] = matrix[i,i] / np.sum(matrix[:,i])
            specificity[i] = (np.sum(np.delete(np.delete(matrix,i,0),i,1))/
                np.sum(np.delete(matrix,1,0)))
        # Optionally average the per-catagory values
        if averaged:
            recall = np.average(recall)
            precision = np.average(precision)
            specificity = np.average(specificity)
        return accuracy, recall, precision, specificity
        
            
        
if __name__ == '__main__':
    
    # Testing the confusion matrix calculations
    (xtrain, ytrain), (xtest, ytest) = tf.keras.datasets.fashion_mnist.load_data()
    from BuildingBlocks import DenseBlock
    block = DenseBlock([784, 196, 196, 49])
    net = SimpleClassifier((28,28), 10, [block])
    net.train(xtrain, ytrain, batch=5000, epochs=20)
    a, r, p, s = net.score(xtest, ytest, averaged = True)