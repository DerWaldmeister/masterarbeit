import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.estimator import regression
from tflearn.layers.merge_ops import merge, merge_outputs


#TODO: Check wether to split convnets
'''
def createCombined2dConvNetNeuralNetworkModelForFutureResourceUtilisation(input_size_states, output_size_actions, learningRate, rowsFutureResourceUtilisationMatrix, columnsFutureResourceUtilisationMatrix):

    # tflearn.init_graph(num_cores=1, gpu_memory_fraction=0.5)

    #### convolutional layers for FutureResourceUtilisationMatrix ####
    # How to configure input_data: https://stackoverflow.com/questions/48482746/tflearn-what-is-input-data
    convnetResourceUtilisation = input_data(shape=[None, rowsFutureResourceUtilisationMatrix, columnsFutureResourceUtilisationMatrix, 1], name='input_futureResourceUtilisationMatrix')

    convnetResourceUtilisation = conv_2d(convnetResourceUtilisation, nb_filter=10, filter_size=3, strides=1, padding='same', activation='relu')
    convnetResourceUtilisation = max_pool_2d(convnetResourceUtilisation, kernel_size=2, strides=2, padding='valid')

    # convnetResourceUtilisation = conv_2d(convnetResourceUtilisation, nb_filter=20, filter_size=3, strides=1, padding='valid', activation='relu')
    # convnetResourceUtilisation = max_pool_2d(convnetResourceUtilisation, kernel_size=2, strides=2, padding='valid')

    convnetResourceUtilisation = flatten(convnetResourceUtilisation)

    # in order to merge network outputs they need to have the same size:
    # fully connected layer for smaller network to have the same size
    convnetResourceUtilisationOutput = fully_connected(convnetResourceUtilisation, n_units=980)
    print("convnetResourceUtilisationOutput: " + str(convnetResourceUtilisationOutput))

    #### convolutional layers for currentState ####
    convnetCurrentState = input_data(shape=[None, input_size_states, input_size_states, 1], name='input_currentState')

    convnetCurrentState = conv_2d(convnetCurrentState, nb_filter=10, filter_size=5, strides=1, padding='same', activation='relu')
    convnetCurrentState = max_pool_2d(convnetCurrentState, kernel_size=2, strides=2, padding='valid')

    convnetCurrentState = conv_2d(convnetCurrentState, nb_filter=20, filter_size=3, strides=1, padding='valid', activation='relu')
    convnetCurrentState = max_pool_2d(convnetCurrentState, kernel_size=2, strides=2, padding='valid')

    convnetCurrentState = flatten(convnetCurrentState)
    convnetCurrentStateOutput = fully_connected(convnetCurrentState, n_units=980)
    print("convnetCurrentStateOutput: " + str(convnetCurrentState))

    # merging the outputs of both convolutional nets
    finalNet = merge([convnetResourceUtilisationOutput, convnetCurrentStateOutput], 'concat', axis=0) # axis=0 is concatenation
    print("finalNet1: " + str(finalNet))
    #### final fully connected layers ####
    #neural_net = input_data(shape=[None, 980])

    finalNet = fully_connected(finalNet, n_units=1800, weights_init='truncated_normal', activation='relu')
    print("finalNet2: " + str(finalNet))
    finalNet = dropout(finalNet, 0.8)
    print("finalNet3: " + str(finalNet))
    finalNet = fully_connected(finalNet, n_units=900, weights_init='truncated_normal', activation='relu')
    print("finalNet4: " + str(finalNet))
    finalNet = dropout(finalNet, 0.8)
    print("finalNet5: " + str(finalNet))
    print("output_size_actions:" + str(output_size_actions))
    finalNet = fully_connected(finalNet, n_units=output_size_actions, activation='softmax')
    print("finalNet6: " + str(finalNet))
    finalNet = regression(finalNet, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy', name='targets')
    print("finalNet7: " + str(finalNet))
    model = tflearn.DNN(finalNet, tensorboard_dir='log')

    return model
'''

def createCombined2dConvNetNeuralNetworkModelForFutureResourceUtilisation(input_size_states, output_size_actions,
                                                                              learningRate, rowsFutureResourceUtilisationMatrix, columnsFutureResourceUtilisationMatrix
                                                                              ):
    # tflearn.init_graph(num_cores=1, gpu_memory_fraction=0.5)

    #### convolutional layers for FutureResourceUtilisationMatrix ####
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

    #### convolutional layers for currentState ####
    convnetCurrentState = input_data(shape=[None, input_size_states, input_size_states, 1], name='input_currentState')

    convnetCurrentState = conv_2d(convnetCurrentState, nb_filter=10, filter_size=5, strides=1, padding='same', activation='relu')
    convnetCurrentState = max_pool_2d(convnetCurrentState, kernel_size=2, strides=2, padding='valid')

    convnetCurrentState = conv_2d(convnetCurrentState, nb_filter=20, filter_size=3, strides=1, padding='valid', activation='relu')
    convnetCurrentState = max_pool_2d(convnetCurrentState, kernel_size=2, strides=2, padding='valid')

    convnetCurrentState = flatten(convnetCurrentState)
    #convnetCurrentState = fully_connected(convnetCurrentState, n_units=980)
    # print("convnetCurrentStateOutput: " + str(convnetCurrentState))

    # merging the outputs of both convolutional nets
    finalNet = merge_outputs([convnetResourceUtilisation, convnetCurrentState], 'concat') # axis=0 is concatenation
    #print("finalNet1: " + str(finalNet))
    #### final fully connected layers ####
    #neural_net = input_data(shape=[None, 980])

    finalNet = fully_connected(finalNet, n_units=1800, weights_init='truncated_normal', activation='relu')
    finalNet = dropout(finalNet, 0.8)

    finalNet = fully_connected(finalNet, n_units=900, weights_init='truncated_normal', activation='relu')
    finalNet = dropout(finalNet, 0.8)

    finalNet = fully_connected(finalNet, n_units=output_size_actions, activation='softmax')
    finalNet = regression(finalNet, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(finalNet, tensorboard_dir='log')

    return model