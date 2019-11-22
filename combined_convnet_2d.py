import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.estimator import regression
from tflearn.layers.merge_ops import merge, merge_outputs
from datetime import datetime



#CONFIGURATION1

def createCombined2dConvNetNeuralNetworkModelForFutureResourceUtilisation(input_size_states, output_size_actions,
                                                                              learningRate, rowsFutureResourceUtilisationMatrix, columnsFutureResourceUtilisationMatrix
                                                                              ):

    # Specify the log directory
    logdir = 'log/2d_combined/' + datetime.now().strftime("%Y%m%d-%H%M%S")

    # tflearn.init_graph(num_cores=1, gpu_memory_fraction=0.5)

    #### convolutional layers for currentState ####
    convnetCurrentState = input_data(shape=[None, input_size_states, input_size_states, 1], name='input_currentState')

    convnetCurrentState = conv_2d(convnetCurrentState, nb_filter=16, filter_size=3, strides=1, padding='valid',
                                  activation='relu')
    convnetCurrentState = max_pool_2d(convnetCurrentState, kernel_size=2, strides=2, padding='valid')

    convnetCurrentState = flatten(convnetCurrentState)


    #### convolutional layers for FutureResourceUtilisationMatrix ####
    # How to configure input_data: https://stackoverflow.com/questions/48482746/tflearn-what-is-input-data
    convnetResourceUtilisation = input_data(
        shape=[None, rowsFutureResourceUtilisationMatrix, columnsFutureResourceUtilisationMatrix, 1],
        name='input_futureResourceUtilisationMatrix')

    convnetResourceUtilisation = conv_2d(convnetResourceUtilisation, nb_filter=16, filter_size=[4, 1], strides=1,
                                         padding='same', activation='relu')
    convnetResourceUtilisation = max_pool_2d(convnetResourceUtilisation, kernel_size=[4, 2], strides=2, padding='valid')

    convnetResourceUtilisation = flatten(convnetResourceUtilisation)

    # merging the outputs of both convolutional nets
    finalNet = merge_outputs([convnetResourceUtilisation, convnetCurrentState], 'concat') # axis=0 is concatenation

    #### final fully connected layers ####
    finalNet = fully_connected(finalNet, n_units=128, weights_init='truncated_normal', activation='relu')
    finalNet = dropout(finalNet, 0.5)

    finalNet = fully_connected(finalNet, n_units=output_size_actions, activation='softmax')
    finalNet = regression(finalNet, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(finalNet, tensorboard_dir=logdir)

    return model