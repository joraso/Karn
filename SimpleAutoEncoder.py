# -*- coding: utf-8 -*-
"""
Created on Thu Feb 9 10:40:23 2021
@author: Joe Raso
"""

import numpy as np
import tensorflow as tf
from utils import connect

class SimpleAutoEncoder():
    """ The simplest possible auto-encoder neural network design. The basic 
        construction is:
        
            input -> encoder -> latent -> decoder -> output
        
        The encoder consists of an input layer, followed by any number of
        interstitial dense hidden layers (specified by 'structure'), then a low
        dimensional latent layer. The decoder archetecture is mirror to the
        encoder. The loss function is the binary cross-entropy between the
        intou and the reconstructed output.
        
        Arguments:
        inputdims   - (int) Dimensions of the input data. Note that this
                      network does not perform any data reshaping.
        latentdims  - (int) Dimension of the latent space.
        blocks      - (list) List of DenseBlock building block objects to
                      construct the (encoding) network from.
                      
        Keywords:
        decode_blocks - (list) List of DenseBlock building block objects to
                      construct the decoding network from. If None (default),
                      the decoder is built as a mirror image of the encoder.
        optimizer   - (str) Keras optimizer to use during training. defaults
                      to 'adam'.
        latent_activation - (str) Activation function to use in the latent 
                      layer. Defaults to 'relu'.
        output_activation - (str) Activation function to use in the output
                      layer.Defaults to 'sigmoid'.
        verbose     - (bool) Toggles command line output. Defaults to True.
        """
    def __init__(self, inputdims, latentdims, blocks, decode_blocks=None,
                 optimizer='adam', latent_activation='relu',
                 output_activation='sigmoid', verbose=True):
        # Note: input does not perform reshaping
        self.inputdims = inputdims
        self.latentdims = latentdims
        self.blocks = blocks
        self.decode_blocks = decode_blocks
        self.optimizer = optimizer
        self.latent_activation = latent_activation
        self.output_activation = output_activation
        self.verbose = verbose
        # Attributes not created by the user
        self.layers = []; self.model = None
        # History tracking
        self.loss=[]; self.accuracy=[]
        # Do the thing:
        self.Build()
        
    def Build(self):
        """Constructs the model from the specified structure."""
        # Define the encoder layers:
        self.input = tf.keras.Input(shape=self.inputdims) 
        self.encoding_layers = []
        for blk in self.blocks:
            self.encoding_layers += blk.layerlist()
        self.latent = tf.keras.layers.Dense(self.latentdims,
                activation = self.latent_activation)
        self.encoding_layers.append(self.latent)
        # Define the decoding layers:
        self.encoded_input = tf.keras.Input(shape=self.latentdims) 
        self.decoding_layers = []
        if self.decode_blocks != None:
            for blk in self.decode_blocks:
                self.decoding_layers += blk.layerlist()
        else:
            for blk in self.blocks:
                self.decoding_layers += blk.layerlist(reverse=True)        
        self.output = tf.keras.layers.Dense(self.inputdims,                                            
                activation = self.output_activation)
        self.decoding_layers.append(self.output)
        # Build the intertwinned models
        self.encoder = tf.keras.Model(self.input,
                connect(self.input, self.encoding_layers))
        self.decoder = tf.keras.Model(self.encoded_input,
                connect(self.encoded_input, self.decoding_layers))
        self.autoencoder = tf.keras.Model(self.input,
                connect(self.input, self.encoding_layers+self.decoding_layers))
        # Compile the auto encoder for training (the intertwined models do
        # not need compiling, since we don't train them individually)
        self.autoencoder.compile(optimizer=self.optimizer,
                metrics=['accuracy'], loss='binary_crossentropy')
                
    def train(self, xtrain, batch=1, epochs=1):
        """ Trains the model on the given data set.
        
            Arguments:
            xtrain - (ndarray) Input training data.
            
            Keywords:
            batch  - (int) Batch size. defaults to 1.
            epochs - (int) Number of epochs. defaults to 1.
            """
        self.autoencoder.fit(xtrain, xtrain, batch_size=batch, epochs=epochs,
                       verbose=self.verbose, shuffle=True)
        self.loss += (self.autoencoder.history.history['loss'])
        self.accuracy += (self.autoencoder.history.history['accuracy'])
    
    def Encode(self, xdata):
        """ Returns the latent space/encoded representation of the data.
        
            Arguments:
            xdata - (ndarray) Input data.
            
            Returns:
            xdata - (ndarray) Encoded representation of the data.
            """
        return self.encoder.predict(xdata)
    def Decode(self, zdata):
        """ Returns the reconstruction of output data based on space/encoded
            representation.
            
            Arguments:
            zdata - (ndarray) Input encoded data.
            
            Returns:
            xdata - (ndarray) Decoded/reconstructed output data.
            """
        return self.decoder.predict(zdata)