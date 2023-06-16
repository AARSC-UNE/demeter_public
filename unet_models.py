#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras import metrics, optimizers, layers, models

def unet(numClasses, patchSize, n_bands, numFilters=32, filterSize=3, lr=0.001, numLayers=5, dropout=0.2, strides=1, dilation_rate=1, 
            padding='SAME', activation='relu', initialiser='glorot_uniform', classActivation='softmax', optimiser='Nadam', 
            loss='categorical_crossentropy'):
    
    # Remove the bottom layer from the number of layers
    numLayers = numLayers-1
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        if numClasses==1:
            classActivation='sigmoid'
            accuracy = metrics.BinaryAccuracy(name='accuracy', dtype=tf.float32)
            #miou = metrics.MeanIoU(num_classes=2, dtype=tf.float32, name='Mean_IoU')
            miou = metrics.BinaryIoU(target_class_ids=[0, 1], dtype=tf.float32, name='Mean_IoU')
        else:
            #miou = MeanIoU(num_classes=numClasses, dtype=tf.float32, name='Mean_IoU')
            miou = metrics.OneHotMeanIoU(num_classes=numClasses, dtype=tf.float32, name='Mean_IoU')
            accuracy = metrics.CategoricalAccuracy(name='accuracy', dtype=tf.float32)
        optimiserFunction = optimizers.get(optimiser)
        optimiserFunction.learning_rate = lr

        inputs = layers.Input((patchSize, patchSize, n_bands))
        #dropout_rate = 1 - keep_prob

        # Encoding path
        connectors = []
        pool = inputs
        for layer in range(numLayers):

            # Calculate the number of filters to use
            filters = 2 ** layer * numFilters

            # Apply convolutional filters
            
            conv = layers.Conv2D(filters=filters, kernel_size=filterSize, strides=strides, dilation_rate=dilation_rate,
                padding=padding, activation=activation, kernel_initializer=initialiser)(pool)
            conv = layers.Conv2D(filters=filters, kernel_size=filterSize, strides=strides, dilation_rate=dilation_rate,
                padding=padding, activation=activation, kernel_initializer=initialiser)(conv)
            conv = layers.BatchNormalization()(conv)
            conv = layers.Dropout(rate=dropout)(conv)
            
            # Keep track of the convolutions for skip connections
            connectors.append(conv)

            # Apply max pooling - reduce resolution
            pool = layers.MaxPool2D(pool_size=2, strides=2, padding=padding)(conv)

        # Bottom of the U-Net

        conv = layers.Conv2D(filters=2 ** numLayers * numFilters, kernel_size=filterSize, strides=strides, dilation_rate=dilation_rate,
                padding=padding, activation=activation, kernel_initializer=initialiser)(pool)
        conv = layers.Conv2D(filters=2 ** numLayers * numFilters, kernel_size=filterSize, strides=strides, dilation_rate=dilation_rate,
                padding=padding, activation=activation, kernel_initializer=initialiser)(conv)
        conv = layers.BatchNormalization()(conv)
        conv = layers.Dropout(rate=dropout)(conv)

        # Decoding path
        for layer in range(numLayers):

            # Calculate where we are on the u-net and get the corresponding conncetion
            connection = numLayers - 1 - layer

            # Calculate the number of filters
            filters = 2 ** connection * numFilters

            up = layers.concatenate([layers.UpSampling2D(size=(2, 2))(conv), connectors[connection]], axis=3)

            # Apply more convolutions. Time no longer exsist here so do the same for both versions
            conv = layers.Conv2D(filters=filters, kernel_size=filterSize, strides=strides, dilation_rate=dilation_rate,
                padding=padding, activation=activation, kernel_initializer=initialiser)(up)
            conv = layers.Conv2D(filters=filters, kernel_size=filterSize, strides=strides, dilation_rate=dilation_rate,
                padding=padding, activation=activation, kernel_initializer=initialiser)(conv)
            conv = layers.BatchNormalization()(conv)
            conv = layers.Dropout(rate=dropout)(conv)
                
        # Produce the final classification
        classification = layers.Conv2D(filters=numClasses, kernel_size=1, activation=classActivation)(conv)

        # Build the model
        model = models.Model(inputs=inputs, outputs=classification, name='u-net')
        # if numClasses==1:
        #      model.compile(optimizer=optimiserFunction, loss=loss, metrics=[accuracy])
        # else:
        #     model.compile(optimizer=optimiserFunction, loss=loss, metrics=[accuracy, miou])
        model.compile(optimizer=optimiserFunction, loss=loss, metrics=[accuracy, miou])# , run_eagerly=False, jit_compile=True)

        return model