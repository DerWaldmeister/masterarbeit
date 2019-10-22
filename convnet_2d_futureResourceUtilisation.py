import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.estimator import regression
from tflearn.layers.merge_ops import merge


#TODO: Check wether to split convnets
def createCombined2dConvNetNeuralNetworkModelForFutureResourceUtilisation(input_size_states, output_size_actions, learningRate, rowsFutureResourceUtilisationMatrix, columnsFutureResourceUtilisationMatrix):


    # tflearn.init_graph(num_cores=1, gpu_memory_fraction=0.5)

    #### convolutional layers for FutureResourceUtilisationMatrix ####
    # How to configure input_data: https://stackoverflow.com/questions/48482746/tflearn-what-is-input-data
    convnetResourceUtilisation = input_data(shape=[None, rowsFutureResourceUtilisationMatrix, columnsFutureResourceUtilisationMatrix, 1], name='input_futureResourceUtilisationMatrix')

    convnetResourceUtilisation = conv_2d(convnetResourceUtilisation, nb_filter=10, filter_size=3, strides=1,padding='same', activation='relu')
    convnetResourceUtilisation = max_pool_2d(convnetResourceUtilisation, kernel_size=2, strides=2, padding='valid')

    # convnetResourceUtilisation = conv_2d(convnetResourceUtilisation, nb_filter=20, filter_size=3, strides=1, padding='valid', activation='relu')
    # convnetResourceUtilisation = max_pool_2d(convnetResourceUtilisation, kernel_size=2, strides=2, padding='valid')

    convnetResourceUtilisationOutput = flatten(convnetResourceUtilisation)

    # in order to merge network outputs they need to have the same size:
    # fully connected layer for smaller network to have the same size
    convnetResourceUtilisationOutput = fully_connected(convnetResourceUtilisationOutput, n_units=980)
    print("convnetResourceUtilisationOutput: " + str(convnetResourceUtilisationOutput))

    #### convolutional layers for currentState ####
    convnetCurrentState = input_data(shape=[None, input_size_states, input_size_states, 1], name='input_currentState')

    convnetCurrentState = conv_2d(convnetCurrentState, nb_filter=10, filter_size=5, strides=1, padding='same', activation='relu')
    convnetCurrentState = max_pool_2d(convnetCurrentState, kernel_size=2, strides=2, padding='valid')

    convnetCurrentState = conv_2d(convnetCurrentState, nb_filter=20, filter_size=3, strides=1, padding='valid', activation='relu')
    convnetCurrentState = max_pool_2d(convnetCurrentState, kernel_size=2, strides=2, padding='valid')

    convnetCurrentStateOutput = flatten(convnetCurrentState)
    print("convnetCurrentStateOutput: " + str(convnetCurrentStateOutput))

    # merging the outputs of both convolutional nets
    combinedOutput = merge([convnetResourceUtilisationOutput, convnetCurrentStateOutput], 'concat', axis=0) # axis=0 is concatenation

    #### final fully connected layers ####
    neuralNet = fully_connected(combinedOutput, n_units=1800, weights_init='truncated_normal', activation='relu')
    neuralNet = dropout(neuralNet, 0.8)

    neuralNet = fully_connected(neuralNet, n_units=900, weights_init='truncated_normal', activation='relu')
    neuralNet = dropout(neuralNet, 0.8)

    neuralNet = fully_connected(neuralNet, n_units=output_size_actions, activation='softmax')
    neuralNet = regression(neuralNet, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(neuralNet, tensorboard_dir='log')

    return model