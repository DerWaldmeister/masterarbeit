import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.estimator import regression
from tflearn.layers.merge_ops import merge


#TODO: Check wether to split convnets
def createCombinedConvNetNeuralNetworkModelForFutureResourceUtilisation(input_size_states, output_size_actions, learningRate, heightFutureResourceUtilisationMatrix, widthFutureResourceUtilisationMatrix):

    # tflearn.init_graph(num_cores=1, gpu_memory_fraction=0.5)

    # convolutional layers for FutureResourceUtilisationMatrix
    convnetResourceUtilisation = input_data(shape=[None, heightFutureResourceUtilisationMatrix, widthFutureResourceUtilisationMatrix,1], name='input')

    convnetResourceUtilisation = conv_2d(convnetResourceUtilisation, nb_filter=10, filter_size=5, strides=1, padding='same', activation='relu')
    convnetResourceUtilisation = max_pool_2d(convnetResourceUtilisation, kernel_size=2, strides=2, padding='valid')

    convnetResourceUtilisation = conv_2d(convnetResourceUtilisation, nb_filter=20, filter_size=3, strides=1, padding='valid', activation='relu')
    convnetResourceUtilisation = max_pool_2d(convnetResourceUtilisation, kernel_size=2, strides=2, padding='valid')

    convnetResourceUtilisationOutput = flatten(convnetResourceUtilisation)

    # convolutional layers for currentState
    convnetCurrentState = input_data(shape=[None, input_size_states, input_size_states, 1], name='input')

    convnetCurrentState = conv_2d(convnetCurrentState, nb_filter=10, filter_size=5, strides=1, padding='same', activation='relu')
    convnetCurrentState = max_pool_2d(convnetCurrentState, kernel_size=2, strides=2, padding='valid')

    convnetCurrentState = conv_2d(convnetCurrentState, nb_filter=20, filter_size=3, strides=1, padding='valid', activation='relu')
    convnetCurrentState = max_pool_2d(convnetCurrentState, kernel_size=2, strides=2, padding='valid')

    convnetCurrentStateOutput = flatten(convnetCurrentState)


    # merging the outputs of both convolutional nets
    mergedOutput = merge(convnetResourceUtilisationOutput, convnetCurrentStateOutput, axis=0) # axis=0 is concatenation

    neuralNet = fully_connected(mergedOutput, n_units=1800, weights_init='truncated_normal', activation='relu')
    neuralNet = dropout(neuralNet, 0.8)

    neuralNet = fully_connected(neuralNet, n_units=900, weights_init='truncated_normal', activation='relu')
    neuralNet = dropout(neuralNet, 0.8)

    neuralNet = fully_connected(neuralNet, n_units=output_size_actions, activation='softmax')
    neuralNet = regression(neuralNet, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy',
                         name='targets')

    model = tflearn.DNN(neuralNet, tensorboard_dir='log')

    return model