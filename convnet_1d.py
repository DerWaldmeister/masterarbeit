import tflearn
from tflearn.layers.conv import conv_1d, max_pool_1d
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.estimator import regression
from datetime import datetime

#1D_conf1_convlay1_fc0  (fromerly: CONFIGURATION simple)
''' 
def create1dConvNetNeuralNetworkModel(input_size, output_size, learningRate):

    # Specify the log directory
    logdir = 'log/1d/' + datetime.now().strftime('%Y%m%d-%H%M%S')
    # Limit gpu memory use
    #tflearn.init_graph(num_cores=1, gpu_memory_fraction=0.7)

    convnet = input_data(shape=[None, input_size], name='input_currentState')
    convnet = tflearn.embedding(convnet, input_dim=input_size, output_dim=2)

    convnet = conv_1d(convnet, nb_filter=16, filter_size=3, strides=1, padding='valid', activation='relu')
    convnet = max_pool_1d(convnet, kernel_size=2, strides=2, padding='valid')

    convnet = flatten(convnet)


    convnet = fully_connected(convnet, n_units=output_size, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy',
                         name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir=logdir)

    return model

'''

#1D_conf2_convlay1_fc1  (formerly: CONFIGURATION 2 )

def create1dConvNetNeuralNetworkModel(input_size, output_size, learningRate):

    # Specify the log directory
    logdir = 'log/1d/' + datetime.now().strftime('%Y%m%d-%H%M%S')
    # Limit gpu memory use
    #tflearn.init_graph(num_cores=1, gpu_memory_fraction=0.7)

    convnet = input_data(shape=[None, input_size], name='input_currentState')
    convnet = tflearn.embedding(convnet, input_dim=input_size, output_dim=2)

    convnet = conv_1d(convnet, nb_filter=16, filter_size=3, strides=1, padding='valid', activation='relu')
    convnet = max_pool_1d(convnet, kernel_size=2, strides=2, padding='valid')

    convnet = flatten(convnet)

    convnet = fully_connected(convnet, n_units=128, weights_init='truncated_normal', activation='relu')
    convnet = dropout(convnet, 0.5)

    convnet = fully_connected(convnet, n_units=output_size, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy',
                         name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir=logdir)

    return model



#1D_conf3_convlay1_fc1  (foremerly: CONFIGURATION 1 )
'''
def create1dConvNetNeuralNetworkModel(input_size, output_size, learningRate):

    # Specify the log directory
    logdir = 'log/1d/' + datetime.now().strftime('%Y%m%d-%H%M%S')
    # Limit gpu memory use
    #tflearn.init_graph(num_cores=1, gpu_memory_fraction=0.7)

    convnet = input_data(shape=[None, input_size], name='input_currentState')
    convnet = tflearn.embedding(convnet, input_dim=input_size, output_dim=2)

    convnet = conv_1d(convnet, nb_filter=16, filter_size=5, strides=1, padding='valid', activation='relu')
    convnet = max_pool_1d(convnet, kernel_size=2, strides=2, padding='valid')

    convnet = flatten(convnet)

    convnet = fully_connected(convnet, n_units=128, weights_init='truncated_normal', activation='relu')
    convnet = dropout(convnet, 0.5)

    convnet = fully_connected(convnet, n_units=output_size, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy',
                         name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir=logdir)

    return model
    '''


#1D_configuration4_convlay1_fc1  (formerly: CONFIGURATION 2a )
'''
def create1dConvNetNeuralNetworkModel(input_size, output_size, learningRate):

    # Specify the log directory
    logdir = 'log/1d/' + datetime.now().strftime('%Y%m%d-%H%M%S')
    # Limit gpu memory use
    #tflearn.init_graph(num_cores=1, gpu_memory_fraction=0.7)

    convnet = input_data(shape=[None, input_size], name='input_currentState')
    convnet = tflearn.embedding(convnet, input_dim=input_size, output_dim=2)

    convnet = conv_1d(convnet, nb_filter=16, filter_size=3, strides=1, padding='valid', activation='relu')
    convnet = max_pool_1d(convnet, kernel_size=2, strides=2, padding='valid')

    convnet = flatten(convnet)

    convnet = fully_connected(convnet, n_units=512, weights_init='truncated_normal', activation='relu')
    convnet = dropout(convnet, 0.5)

    convnet = fully_connected(convnet, n_units=output_size, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy',
                         name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir=logdir)

    return model
'''

#1D_conf5_convlay1_fc1  (formerly: CONFIGURATION 2b)
'''
def create1dConvNetNeuralNetworkModel(input_size, output_size, learningRate):

    # Specify the log directory
    logdir = 'log/1d/' + datetime.now().strftime('%Y%m%d-%H%M%S')
    # Limit gpu memory use
    #tflearn.init_graph(num_cores=1, gpu_memory_fraction=0.7)

    convnet = input_data(shape=[None, input_size], name='input_currentState')
    convnet = tflearn.embedding(convnet, input_dim=input_size, output_dim=2)

    convnet = conv_1d(convnet, nb_filter=16, filter_size=3, strides=1, padding='valid', activation='relu')
    convnet = max_pool_1d(convnet, kernel_size=2, strides=2, padding='valid')

    convnet = flatten(convnet)

    convnet = fully_connected(convnet, n_units=32, weights_init='truncated_normal', activation='relu')
    convnet = dropout(convnet, 0.5)

    convnet = fully_connected(convnet, n_units=output_size, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy',
                         name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir=logdir)

    return model
'''


# 1D_conf6_convlay3_fc2  (formerly: CONFIGURATION 5)
'''
def create1dConvNetNeuralNetworkModel(input_size, output_size, learningRate):

    # Specify the log directory
    logdir = 'log/1d/' + datetime.now().strftime('%Y%m%d-%H%M%S')
    # Limit gpu memory use
    #tflearn.init_graph(num_cores=1, gpu_memory_fraction=0.7)

    convnet = input_data(shape=[None, input_size], name='input_currentState')
    convnet = tflearn.embedding(convnet, input_dim=input_size, output_dim=2)

    convnet = conv_1d(convnet, nb_filter=16, filter_size=5, strides=1, padding='valid', activation='relu')
    convnet = max_pool_1d(convnet, kernel_size=2, strides=2, padding='valid')

    convnet = conv_1d(convnet, nb_filter=16, filter_size=3, strides=1, padding='valid', activation='relu')
    convnet = max_pool_1d(convnet, kernel_size=2, strides=2, padding='valid')

    convnet = conv_1d(convnet, nb_filter=32, filter_size=3, strides=1, padding='valid', activation='relu')
    convnet = max_pool_1d(convnet, kernel_size=2, strides=2, padding='valid')

    convnet = flatten(convnet)

    convnet = fully_connected(convnet, n_units=256, weights_init='truncated_normal', activation='relu')
    convnet = dropout(convnet, 0.5)

    convnet = fully_connected(convnet, n_units=128, weights_init='truncated_normal', activation='relu')
    convnet = dropout(convnet, 0.5)

    convnet = fully_connected(convnet, n_units=output_size, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy',
                         name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir=logdir)

    return model
'''










################### Archiv - Configurations #####################

#CONFIGURATION 3
'''
def create1dConvNetNeuralNetworkModel(input_size, output_size, learningRate):

    # Specify the log directory
    logdir = 'log/1d/' + datetime.now().strftime('%Y%m%d-%H%M%S')
    # Limit gpu memory use
    #tflearn.init_graph(num_cores=1, gpu_memory_fraction=0.7)

    convnet = input_data(shape=[None, input_size], name='input_currentState')
    convnet = tflearn.embedding(convnet, input_dim=input_size, output_dim=2)

    convnet = conv_1d(convnet, nb_filter=16, filter_size=3, strides=1, padding='valid', activation='relu')
    convnet = max_pool_1d(convnet, kernel_size=2, strides=2, padding='valid')

    convnet = conv_1d(convnet, nb_filter=32, filter_size=3, strides=1, padding='valid', activation='relu')
    convnet = max_pool_1d(convnet, kernel_size=2, strides=2, padding='valid')

    convnet = flatten(convnet)

    convnet = fully_connected(convnet, n_units=512, weights_init='truncated_normal', activation='relu')
    convnet = dropout(convnet, 0.5)

    convnet = fully_connected(convnet, n_units=output_size, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy',
                         name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir=logdir)

    return model
'''

#CONFIGURATION 3a
'''
def create1dConvNetNeuralNetworkModel(input_size, output_size, learningRate):

    # Specify the log directory
    logdir = 'log/1d/' + datetime.now().strftime('%Y%m%d-%H%M%S')
    # Limit gpu memory use
    #tflearn.init_graph(num_cores=1, gpu_memory_fraction=0.7)

    convnet = input_data(shape=[None, input_size], name='input_currentState')
    convnet = tflearn.embedding(convnet, input_dim=input_size, output_dim=2)

    convnet = conv_1d(convnet, nb_filter=16, filter_size=5, strides=1, padding='valid', activation='relu')
    convnet = max_pool_1d(convnet, kernel_size=2, strides=2, padding='valid')

    convnet = conv_1d(convnet, nb_filter=32, filter_size=3, strides=1, padding='valid', activation='relu')
    convnet = max_pool_1d(convnet, kernel_size=2, strides=2, padding='valid')

    convnet = flatten(convnet)

    convnet = fully_connected(convnet, n_units=512, weights_init='truncated_normal', activation='relu')
    convnet = dropout(convnet, 0.5)

    convnet = fully_connected(convnet, n_units=output_size, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy',
                         name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir=logdir)

    return model
'''


#CONFIGURATION 4
'''
def create1dConvNetNeuralNetworkModel(input_size, output_size, learningRate):

    # Specify the log directory
    logdir = 'log/1d/' + datetime.now().strftime('%Y%m%d-%H%M%S')
    # Limit gpu memory use
    #tflearn.init_graph(num_cores=1, gpu_memory_fraction=0.7)

    convnet = input_data(shape=[None, input_size], name='input_currentState')
    convnet = tflearn.embedding(convnet, input_dim=input_size, output_dim=2)

    convnet = conv_1d(convnet, nb_filter=16, filter_size=3, strides=1, padding='valid', activation='relu')
    convnet = max_pool_1d(convnet, kernel_size=2, strides=2, padding='valid')

    convnet = conv_1d(convnet, nb_filter=32, filter_size=3, strides=1, padding='valid', activation='relu')
    convnet = max_pool_1d(convnet, kernel_size=2, strides=2, padding='valid')

    convnet = flatten(convnet)

    convnet = fully_connected(convnet, n_units=256, weights_init='truncated_normal', activation='relu')
    convnet = dropout(convnet, 0.5)

    convnet = fully_connected(convnet, n_units=128, weights_init='truncated_normal', activation='relu')
    convnet = dropout(convnet, 0.5)

    convnet = fully_connected(convnet, n_units=output_size, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy',
                         name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir=logdir)

    return model
  '''

