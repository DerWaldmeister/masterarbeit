import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.estimator import regression
from datetime import datetime


#2D_conf_convlay1_fc1     (formerly:  CONFIGURATION 1)
'''
def create2dConvNetNeuralNetworkModel(input_size, output_size, learningRate):

    # Specify the log directory
    logdir = 'log/2d/' + datetime.now().strftime('%Y%m%d-%H%M%S')

    convnet = input_data(shape=[None, input_size, input_size,1], name='input_currentState')

    #tflearn.init_graph(num_cores=1, gpu_memory_fraction=0.8)

    convnet = conv_2d(convnet, nb_filter=16, filter_size=3, strides=1, padding='valid', activation='relu')
    convnet = max_pool_2d(convnet, kernel_size=2, strides=2, padding='valid')

    convnet = flatten(convnet)

    convnet = fully_connected(convnet, n_units=128, weights_init='truncated_normal', activation='relu')
    convnet = dropout(convnet, 0.5)

    convnet = fully_connected(convnet, n_units=output_size, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir=logdir)

    return model
'''

#2D_conf2_convlay1_fc1    (formerly:  CONFIGURATION 2)
'''
def create2dConvNetNeuralNetworkModel(input_size, output_size, learningRate):

    # Specify the log directory
    logdir = 'log/2d/' + datetime.now().strftime('%Y%m%d-%H%M%S')

    convnet = input_data(shape=[None, input_size, input_size,1], name='input_currentState')

    tflearn.init_graph(num_cores=1, gpu_memory_fraction=0.8)

    convnet = conv_2d(convnet, nb_filter=16, filter_size=3, strides=1, padding='valid', activation='relu')
    convnet = max_pool_2d(convnet, kernel_size=2, strides=2, padding='valid')

    convnet = flatten(convnet)

    convnet = fully_connected(convnet, n_units=256, weights_init='truncated_normal', activation='relu')
    convnet = dropout(convnet, 0.5)

    convnet = fully_connected(convnet, n_units=output_size, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir=logdir)

    return model
'''

#2D_conf3_convlay1_fc1    (formerly:   CONFIGURATION 3)
'''
def create2dConvNetNeuralNetworkModel(input_size, output_size, learningRate):

    # Specify the log directory
    logdir = 'log/2d/' + datetime.now().strftime('%Y%m%d-%H%M%S')

    convnet = input_data(shape=[None, input_size, input_size,1], name='input_currentState')

    #tflearn.init_graph(num_cores=1, gpu_memory_fraction=0.8)

    convnet = conv_2d(convnet, nb_filter=64, filter_size=3, strides=1, padding='valid', activation='relu')
    convnet = max_pool_2d(convnet, kernel_size=2, strides=2, padding='valid')

    convnet = flatten(convnet)

    convnet = fully_connected(convnet, n_units=256, weights_init='truncated_normal', activation='relu')
    convnet = dropout(convnet, 0.5)

    convnet = fully_connected(convnet, n_units=output_size, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir=logdir)

    return model
'''


#2D_conf4_convlay2_fc1      (formerly:  CONFIGURATION 4a)
'''
def create2dConvNetNeuralNetworkModel(input_size, output_size, learningRate):

    # Specify the log directory
    logdir = 'log/2d/' + datetime.now().strftime('%Y%m%d-%H%M%S')

    convnet = input_data(shape=[None, input_size, input_size,1], name='input_currentState')

    #tflearn.init_graph(num_cores=1, gpu_memory_fraction=0.9)

    convnet = conv_2d(convnet, nb_filter=16, filter_size=3, strides=1, padding='valid', activation='relu')
    convnet = max_pool_2d(convnet, kernel_size=2, strides=2, padding='valid')

    convnet = conv_2d(convnet, nb_filter=32, filter_size=3, strides=1, padding='valid', activation='relu')
    convnet = max_pool_2d(convnet, kernel_size=2, strides=2, padding='valid')

    convnet = flatten(convnet)

    convnet = fully_connected(convnet, n_units=256, weights_init='truncated_normal', activation='relu')
    convnet = dropout(convnet, 0.5)

    convnet = fully_connected(convnet, n_units=output_size, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir=logdir)

    return model
    
'''

#2D_conf5_convlay2_fc1     (formerly:   CONFIGURATION 5)
'''
def create2dConvNetNeuralNetworkModel(input_size, output_size, learningRate):

    # Specify the log directory
    logdir = 'log/2d/' + datetime.now().strftime('%Y%m%d-%H%M%S')

    convnet = input_data(shape=[None, input_size, input_size,1], name='input_currentState')

    #tflearn.init_graph(num_cores=1, gpu_memory_fraction=0.9)

    convnet = conv_2d(convnet, nb_filter=16, filter_size=5, strides=1, padding='valid', activation='relu')
    convnet = max_pool_2d(convnet, kernel_size=2, strides=2, padding='valid')

    convnet = conv_2d(convnet, nb_filter=32, filter_size=3, strides=1, padding='valid', activation='relu')
    convnet = max_pool_2d(convnet, kernel_size=2, strides=2, padding='valid')

    convnet = flatten(convnet)

    convnet = fully_connected(convnet, n_units=256, weights_init='truncated_normal', activation='relu')
    convnet = dropout(convnet, 0.5)

    convnet = fully_connected(convnet, n_units=output_size, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir=logdir)

    return model
'''

#2D_conf6_convlay2_fc2    (formerly:   CONFIGURATION 6)

def create2dConvNetNeuralNetworkModel(input_size, output_size, learningRate):

    # Specify the log directory
    logdir = 'log/2d/' + datetime.now().strftime('%Y%m%d-%H%M%S')

    convnet = input_data(shape=[None, input_size, input_size,1], name='input_currentState')

    tflearn.init_graph(num_cores=1, gpu_memory_fraction=0.9)

    convnet = conv_2d(convnet, nb_filter=16, filter_size=5, strides=1, padding='valid', activation='relu')
    convnet = max_pool_2d(convnet, kernel_size=2, strides=2, padding='valid')

    convnet = conv_2d(convnet, nb_filter=32, filter_size=3, strides=1, padding='valid', activation='relu')
    convnet = max_pool_2d(convnet, kernel_size=2, strides=2, padding='valid')

    convnet = flatten(convnet)

    convnet = fully_connected(convnet, n_units=256, weights_init='truncated_normal', activation='relu')
    convnet = dropout(convnet, 0.5)

    convnet = fully_connected(convnet, n_units=128, weights_init='truncated_normal', activation='relu')
    convnet = dropout(convnet, 0.5)

    convnet = fully_connected(convnet, n_units=output_size, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir=logdir)

    return model


#2D_conf7_convlay2_fc2   (formerly:   CONFIGURATION 7)
'''
def create2dConvNetNeuralNetworkModel(input_size, output_size, learningRate):

    # Specify the log directory
    logdir = 'log/2d/' + datetime.now().strftime('%Y%m%d-%H%M%S')

    convnet = input_data(shape=[None, input_size, input_size,1], name='input_currentState')

    tflearn.init_graph(num_cores=1, gpu_memory_fraction=0.9)

    convnet = conv_2d(convnet, nb_filter=16, filter_size=5, strides=1, padding='valid', activation='relu')
    convnet = max_pool_2d(convnet, kernel_size=2, strides=2, padding='valid')

    convnet = conv_2d(convnet, nb_filter=32, filter_size=3, strides=1, padding='valid', activation='relu')
    convnet = max_pool_2d(convnet, kernel_size=2, strides=2, padding='valid')

    convnet = flatten(convnet)

    convnet = fully_connected(convnet, n_units=512, weights_init='truncated_normal', activation='relu')
    convnet = dropout(convnet, 0.5)

    convnet = fully_connected(convnet, n_units=256, weights_init='truncated_normal', activation='relu')
    convnet = dropout(convnet, 0.5)

    convnet = fully_connected(convnet, n_units=output_size, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir=logdir)

    return model
'''

#2D_conf8_convlay2_fc2     (formerly:   CONFIGURATION 8)
'''
def create2dConvNetNeuralNetworkModel(input_size, output_size, learningRate):

    # Specify the log directory
    logdir = 'log/2d/' + datetime.now().strftime('%Y%m%d-%H%M%S')

    convnet = input_data(shape=[None, input_size, input_size,1], name='input_currentState')

    tflearn.init_graph(num_cores=1, gpu_memory_fraction=0.9)

    convnet = conv_2d(convnet, nb_filter=16, filter_size=5, strides=1, padding='valid', activation='relu')
    convnet = max_pool_2d(convnet, kernel_size=2, strides=2, padding='valid')

    convnet = conv_2d(convnet, nb_filter=32, filter_size=3, strides=1, padding='valid', activation='relu')
    convnet = max_pool_2d(convnet, kernel_size=2, strides=2, padding='valid')

    convnet = flatten(convnet)

    convnet = fully_connected(convnet, n_units=1024, weights_init='truncated_normal', activation='relu')
    convnet = dropout(convnet, 0.5)

    convnet = fully_connected(convnet, n_units=512, weights_init='truncated_normal', activation='relu')
    convnet = dropout(convnet, 0.5)

    convnet = fully_connected(convnet, n_units=output_size, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir=logdir)

    return model
'''

#2D_conf9_convlay3_fc2    (formerly:  CONFIGURATION 9)
'''
def create2dConvNetNeuralNetworkModel(input_size, output_size, learningRate):

    # Specify the log directory
    logdir = 'log/2d/' + datetime.now().strftime('%Y%m%d-%H%M%S')

    convnet = input_data(shape=[None, input_size, input_size,1], name='input_currentState')

    #tflearn.init_graph(num_cores=1, gpu_memory_fraction=0.9)

    convnet = conv_2d(convnet, nb_filter=16, filter_size=5, strides=1, padding='valid', activation='relu')
    convnet = max_pool_2d(convnet, kernel_size=2, strides=2, padding='valid')

    convnet = conv_2d(convnet, nb_filter=32, filter_size=3, strides=1, padding='valid', activation='relu')
    convnet = max_pool_2d(convnet, kernel_size=2, strides=2, padding='valid')

    convnet = conv_2d(convnet, nb_filter=64, filter_size=3, strides=1, padding='valid', activation='relu')
    convnet = max_pool_2d(convnet, kernel_size=2, strides=2, padding='valid')

    convnet = flatten(convnet)

    convnet = fully_connected(convnet, n_units=1024, weights_init='truncated_normal', activation='relu')
    convnet = dropout(convnet, 0.5)

    convnet = fully_connected(convnet, n_units=512, weights_init='truncated_normal', activation='relu')
    convnet = dropout(convnet, 0.5)

    convnet = fully_connected(convnet, n_units=output_size, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir=logdir)

    return model
    
'''




################### Archiv - Configurations #####################




#CONFIGURATION 4
'''
def create2dConvNetNeuralNetworkModel(input_size, output_size, learningRate):

    # Specify the log directory
    logdir = 'log/2d/' + datetime.now().strftime('%Y%m%d-%H%M%S')

    convnet = input_data(shape=[None, input_size, input_size,1], name='input_currentState')

    tflearn.init_graph(num_cores=1, gpu_memory_fraction=0.9)

    convnet = conv_2d(convnet, nb_filter=64, filter_size=3, strides=1, padding='valid', activation='relu')
    convnet = max_pool_2d(convnet, kernel_size=2, strides=2, padding='valid')

    convnet = conv_2d(convnet, nb_filter=128, filter_size=3, strides=1, padding='valid', activation='relu')
    convnet = max_pool_2d(convnet, kernel_size=2, strides=2, padding='valid')

    convnet = flatten(convnet)

    convnet = fully_connected(convnet, n_units=256, weights_init='truncated_normal', activation='relu')
    convnet = dropout(convnet, 0.5)

    convnet = fully_connected(convnet, n_units=output_size, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir=logdir)

    return model
'''


#CONFIGURATION 10
'''
def create2dConvNetNeuralNetworkModel(input_size, output_size, learningRate):

    # Specify the log directory
    logdir = 'log/2d/' + datetime.now().strftime('%Y%m%d-%H%M%S')

    convnet = input_data(shape=[None, input_size, input_size,1], name='input_currentState')

    #tflearn.init_graph(num_cores=1, gpu_memory_fraction=0.9)

    convnet = conv_2d(convnet, nb_filter=16, filter_size=5, strides=1, padding='valid', activation='relu')
    convnet = max_pool_2d(convnet, kernel_size=2, strides=2, padding='valid')

    convnet = conv_2d(convnet, nb_filter=32, filter_size=3, strides=1, padding='valid', activation='relu')
    convnet = max_pool_2d(convnet, kernel_size=2, strides=2, padding='valid')

    convnet = conv_2d(convnet, nb_filter=64, filter_size=3, strides=1, padding='valid', activation='relu')
    convnet = max_pool_2d(convnet, kernel_size=2, strides=2, padding='valid')

    convnet = flatten(convnet)

    convnet = fully_connected(convnet, n_units=1024, weights_init='truncated_normal', activation='relu')
    convnet = dropout(convnet, 0.5)

    convnet = fully_connected(convnet, n_units=1024, weights_init='truncated_normal', activation='relu')
    convnet = dropout(convnet, 0.5)

    convnet = fully_connected(convnet, n_units=output_size, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir=logdir)

    return model
    '''
