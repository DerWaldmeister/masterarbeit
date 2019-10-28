import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.conv import conv_1d, max_pool_1d
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.estimator import regression
from tflearn.layers.merge_ops import merge, merge_outputs




def createCombined1dConvNetNeuralNetworkModelForFutureResourceUtilisation(input_size_states, output_size_actions,
                                                                              learningRate, rowsFutureResourceUtilisationMatrix, columnsFutureResourceUtilisationMatrix
                                                                              ):
    # tflearn.init_graph(num_cores=1, gpu_memory_fraction=0.5)

    #### 2d-convolutional layers for FutureResourceUtilisationMatrix ####
    # How to configure input_data: https://stackoverflow.com/questions/48482746/tflearn-what-is-input-data
    convnetResourceUtilisation = input_data(shape=[None, rowsFutureResourceUtilisationMatrix, columnsFutureResourceUtilisationMatrix, 1], name='input_futureResourceUtilisationMatrix')

    convnetResourceUtilisation = conv_2d(convnetResourceUtilisation, nb_filter=10, filter_size=3, strides=1, padding='same', activation='relu')
    convnetResourceUtilisation = max_pool_2d(convnetResourceUtilisation, kernel_size=2, strides=2, padding='valid')

    #convnetResourceUtilisation = conv_2d(convnetResourceUtilisation, nb_filter=20, filter_size=3, strides=1, padding='valid', activation='relu')
    #convnetResourceUtilisation = max_pool_2d(convnetResourceUtilisation, kernel_size=2, strides=2, padding='valid')

    convnetResourceUtilisation = flatten(convnetResourceUtilisation)

    # in order to merge network outputs they need to have the same size:
    # fully connected layer for smaller network to have the same size
    #convnetResourceUtilisation = fully_connected(convnetResourceUtilisation, n_units=980)
    #print("convnetResourceUtilisationOutput: " + str(convnetResourceUtilisationOutput))

    #### 1d-convolutional layers for currentState ####
    convnetCurrentState = input_data(shape=[None, input_size_states], name='input_currentState')
    convnetCurrentState = tflearn.embedding(convnetCurrentState, input_dim=input_size_states, output_dim=2)

    convnetCurrentState = conv_1d(convnetCurrentState, nb_filter=10, filter_size=5, strides=1, padding='same', activation='relu')
    convnetCurrentState = max_pool_1d(convnetCurrentState, kernel_size=2, strides=2, padding='valid')

    convnetCurrentState = conv_1d(convnetCurrentState, nb_filter=20, filter_size=3, strides=1, padding='valid', activation='relu')
    convnetCurrentState = max_pool_1d(convnetCurrentState, kernel_size=2, strides=2, padding='valid')

    convnetCurrentState = flatten(convnetCurrentState)

    # merging the outputs of both convolutional nets
    finalNet = merge_outputs([convnetResourceUtilisation, convnetCurrentState], 'concat') # axis=0 is concatenation


    #### final fully connected layers ####
    finalNet = fully_connected(finalNet, n_units=1800, weights_init='truncated_normal', activation='relu')
    finalNet = dropout(finalNet, 0.8)

    finalNet = fully_connected(finalNet, n_units=900, weights_init='truncated_normal', activation='relu')
    finalNet = dropout(finalNet, 0.8)

    finalNet = fully_connected(finalNet, n_units=output_size_actions, activation='softmax')
    finalNet = regression(finalNet, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(finalNet, tensorboard_dir='log')

    return model