from .distance import absolute_distance

import time
import numpy as np

from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, Lambda, Flatten, Dense
# from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall

def initialize_bias(shape, name=None, dtype=None):
    return np.random.normal(loc=0.5, scale=1e-2, size=shape)
  
def initialize_weights(shape, name=None, dtype=None):
    return np.random.normal(loc=0.0, scale=1e-2, size=shape)
  
def initialize_weights_dense(shape, name=None, dtype=None):
    return np.random.normal(loc=0.0, scale=0.2, size=shape)
  
def siamese_model(
    input_shape=(250, 250, 1), 
    filters=64, 
    kernel_initializer=initialize_weights,
    kernel_initializer_d=initialize_weights_dense,
    kernel_regularizer=l2(2e-4),
    kernel_regularizer_d=l2(1e-3),
    bias_initializer=initialize_bias,
    kernel_size_list=[(10, 10), (7, 7), (4, 4), (4, 4)],
    units=64*64, # filters*64
    optimizer=Adam(lr=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy', Precision(name='Precision'), Recall(name='Recall')],
    pretrained_weights=None,
    model_path=None,
    distance=absolute_distance,
    distance_output_shape=None,
    prediction_activation='sigmoid',
):
    if model_path is not None:
        return load_model(model_path)
#     Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    

    model = Sequential()
    
    # Convolutional Layer 1
    model.add(
        Conv2D(
            filters=filters, 
            kernel_size=kernel_size_list[0], 
            activation='relu', 
            input_shape=input_shape,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer
        )
    )
    model.add(MaxPooling2D())
#     model.add(
#         BatchNormalization(
#             axis=1
# #             axis=-1,
# #             momentum=0.99,
# #             epsilon=0.001,
# #             center=True,
# #             scale=True,
# #             beta_initializer='zeros',
# #             gamma_initializer='ones',
# #             moving_mean_initializer='zeros',
# #             moving_variance_initializer='ones',
# #             beta_regularizer=None,
# #             gamma_regularizer=None,
# #             beta_constraint=None,
# #             gamma_constraint=None,
# #             renorm=False,
# #             renorm_clipping=None,
# #             renorm_momentum=0.99,
# #             fused=None,
# #             trainable=True,
# #             virtual_batch_size=None,
# #             adjustment=None,
# #             name=None,
# #             **kwargs
#         )
#     )
    
    # Convolutional Layer 2
    model.add(
        Conv2D(
            filters=filters*2, 
            kernel_size=kernel_size_list[1], 
            activation='relu',
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer
        )
    )
    model.add(MaxPooling2D())   
#     model.add(BatchNormalization(axis=1))
    
    # Convolutional Layer 3
    model.add(
        Conv2D(
            filters=filters*4, 
            kernel_size=kernel_size_list[2], 
            activation='relu', 
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer, 
            kernel_regularizer=kernel_regularizer
        )
    )
    model.add(MaxPooling2D())
#     model.add(BatchNormalization(axis=1))    
    
    # Convolutional Layer 4
    model.add(
        Conv2D(
            filters=filters*8, 
            kernel_size=kernel_size_list[3], 
            activation='relu', 
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer, 
            kernel_regularizer=kernel_regularizer
        )
    )
#     model.add(BatchNormalization(axis=1))    
    
    # Flatten Layer
    model.add(Flatten())
    model.add(
        Dense(
            units=units,
            activation='sigmoid',
            kernel_regularizer=kernel_regularizer_d,
            kernel_initializer=kernel_initializer_d,
            bias_initializer=bias_initializer
        )
    )
    
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    
    # Add a customized layer to compute the absolute difference between the encodings
    L1_distance = Lambda(distance, distance_output_shape)([encoded_l, encoded_r])
    
    # Add a dense layer with a sigmoid unit to generate the similarity score
#     prediction = Dense(1, activation=prediction_activation, bias_initializer=bias_initializer)(L1_distance)
    prediction = L1_distance
  
    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)
    
    siamese_net.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    #siamese_net.summary()
       
    if(pretrained_weights):
        siamese_net.load_weights(pretrained_weights)
    
    # return the model
    return siamese_net