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
        
class Conv2Dblock:
    """ An object containing the parameters of a compound convolutional layer,
        consisting of a Conv2D layer + a 2D pooling layer.
        
        Arguments:
        filters     - (int) The dimensionality of the output space of the
                      convulutional piece of the block.
        kernel_size - (tuple) The kernal size of the convolution.
        
        Keywords:
        activation  - (str) Activation function to use in the convolutional
                      layers. Defauls to 'relu'.
        pooltype    - (str) The type of pooling to be performed in the pooling
                      layer. Current options are: 
                          'max' - Max pooling
                          'ave' - average pooling
                      Defaults to 'max'.
        pool_size   - (tuple) The (vertical, horizontal) factors by which to
                      downscale in the pooling layer. Defaults to (2,2).
        conv_params - (dict) Dictionary of keyword parameters to pass to the
                      convolutional layer (if different from keras defaults).
        pool_params - (dict) Dictionary of keyword parameters to pass to the
                      average or max pooling layer (if different from keras
                      defaults).
        ups_params  - (dict) Dictionary of keyword parameters to pass to the
                      up-sampling layer in inverted layers (if different from
                      keras defaults).
        """
    def __init__(self, filters, kernel_size, activation='relu', pooltype='max',
                 pool_size=(2,2), conv_params={}, pool_params={}):
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.pooltype = pooltype
        if self.pooltype not in ['max', 'ave']:
            print("Error: unrecognized pooling type.")
        self.pool_size = pool_size
        self.conv_params = {'strides':(1, 1), 'padding':"valid",
            'data_format':'channels_last',
            'kernel_initializer':"glorot_uniform",
            'bias_initializer':"zeros",
            'dilation_rate':(1, 1), 'groups':1, 'use_bias':True,
            'kernel_regularizer':None, 'bias_regularizer':None,
            'activity_regularizer':None, 'kernel_constraint':None, 
            'bias_constraint':None}
        self.conv_params.update(conv_params)
        self.pool_params = {'strides':None, 'padding':"valid",
            'data_format':'channels_last'}
        self.pool_params.update(pool_params)
        self.ups_params = {'interpolation':"nearest",
            'data_format':'channels_last'}
        self.ups_params.update(pool_params)

    def layerlist(self, inverted=False):
        """ Conscructs the keras layers comprising the convolutional block and
            returns a (2-member) list to be compiled by the nework object.
            
            Keywords:
            inverted - (bool) If True, returns an iverted convolution (up-
                       sampling followed by deconvolution). Defaults to False.
                       
            Returns:
            layers   - (list) A list of un-built keras layers.
            """
        
        if inverted:
            deconvlayer = tf.keras.layers.Conv2DTranspose(self.filters,
                self.kernel_size, activation = self.activation,
                **self.conv_params)
            uppool = tf.keras.layers.UpSampling2D(size=self.pool_size,
                **self.ups_params)
            return [uppool, deconvlayer]
        else:
            convlayer = tf.keras.layers.Conv2D(self.filters, self.kernel_size,
                activation = self.activation, **self.conv_params)
            if self.pooltype == 'max':
                pool = tf.keras.layers.MaxPooling2D(pool_size=self.pool_size,
                    **self.pool_params)
            elif self.pooltype == 'ave':
                pool = tf.keras.layers.AveragePooling2D(
                    pool_size=self.pool_size, **self.pool_params)
            return [convlayer, pool]
        
    def output_shape(self, input_shape, inverted=False):
        """ Attempts to predict the output tensor shape of the block. (not
            including the channel dimension.)
        
            Arguments:
            input_shape  - (tuple, len=3) The input shape anitcipated from the
                           previous layer/block.
        
            Returns:
            output_shape - (tuple, len=3) The predicted output dimensions of
                           the convolution block.
            """
        def reduce(n, c, p):
            d = n + 1 - c # dimension reduction from convolution
            d = int(d/p) # reduction from pooling
            return d
        def expand(n, c, p):
            d = int(n*p) #  increase from upsampling
            d = d - 1 + c # increase from de-convolution
            return d
        change = expand if inverted else reduce
        x = change(input_shape[0], self.kernel_size[0],
            self.pool_size[0])
        y = change(input_shape[1], self.kernel_size[1],
            self.pool_size[1])
        return (x, y, self.filters)
            