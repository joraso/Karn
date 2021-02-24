# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 12:58:17 2021
@author: Joe Raso
"""

import tensorflow as tf

class DenseBlock:
    """ Object containing the parameters for a block of hidden dense layers.
        
        Arguments:
        structure    - (list) List of ints specifying the dimensionality 
                       (number of neurons) of each hidden layer in the block.
        
        Keywords:
        activation   - (str) Activation function to use in the hidden layers.
                       Defauls to 'relu'.
        dense_params - (dict) Dictionary of keyword arguments to pass to the
                       hidden layers (if different from keras defaults).
        """
    def __init__(self, structure, activation='relu', dense_params={}):
        self.structure = structure
        self.activation = activation
        self.dense_params = {'use_bias':True,
            'kernel_initializer':"glorot_uniform",
            'bias_initializer':"zeros",
            'kernel_regularizer':None, 'bias_regularizer':None,
            'activity_regularizer':None, 'kernel_constraint':None, 
            'bias_constraint':None}
        self.dense_params.update(dense_params)
    
    def layerlist(self, reverse=False):
        """ Consructs the keras layers comprising the block hidden layers and
            returns them as list to be compiled by the nework object.
            
            Keywords:
            reverse   - (bool) If True, returns a list of layers in the reverse
                        order. Defaults to False.
            """
        layers = []; structure = self.structure.copy()
        # processing the 'reverse' keyword
        if reverse:
            structure.reverse()
        # Building the list
        for i in structure:
            layers.append(tf.keras.layers.Dense(i,
                activation = self.activation, **self.dense_params))
        return layers
        