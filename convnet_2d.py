import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.estimator import regression
from datetime import datetime


# CONFIGURATION X  - sentdex tutorial inspired
'''
def create2dConvNetNeuralNetworkModel(input_size, output_size, learningRate):

    convnet = input_data(shape=[None, input_size, input_size,1], name='input')

    tflearn.init_graph(num_cores=1, gpu_memory_fraction=0.5)

    convnet = conv_2d(convnet, 32, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 64, 2, activation='relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, output_size, activation='softmax')
    # Regression layer is used for backpropagation
    convnet = regression(convnet, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy',
                         name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir="log")

    return model

'''
'''
#CONFIGURATION 1 - 1st own model - standard configuration
def create2dConvNetNeuralNetworkModel(input_size, output_size, learningRate):

    convnet = input_data(shape=[None, input_size, input_size,1], name='input')

    #tflearn.init_graph(num_cores=1, gpu_memory_fraction=0.5)

    convnet = conv_2d(convnet, nb_filter=10, filter_size=5, strides=1, padding='same', activation='relu')
    convnet = max_pool_2d(convnet, kernel_size=2, strides=1, padding='valid')

    convnet = conv_2d(convnet, nb_filter=20, filter_size=3, strides=1, padding='valid', activation='relu')
    convnet = max_pool_2d(convnet, kernel_size=2, strides=1, padding='valid')

    convnet = flatten(convnet)

    convnet = fully_connected(convnet, n_units=1800, weights_init='truncated_normal', activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, n_units=900, weights_init='truncated_normal', activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, n_units=output_size, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    return model



'''
#CONFIGURATION 2
'''
def create2dConvNetNeuralNetworkModel(input_size, output_size, learningRate):

    convnet = input_data(shape=[None, input_size, input_size,1], name='input')

    #tflearn.init_graph(num_cores=1, gpu_memory_fraction=0.7)

    convnet = conv_2d(convnet, nb_filter=10, filter_size=7, strides=1, padding='same', activation='relu')
    convnet = max_pool_2d(convnet, kernel_size=2, strides=1, padding='valid')

    convnet = conv_2d(convnet, nb_filter=20, filter_size=5, strides=1, padding='valid', activation='relu')
    convnet = max_pool_2d(convnet, kernel_size=2, strides=1, padding='valid')

    convnet = conv_2d(convnet, nb_filter=30, filter_size=3, strides=1, padding='valid', activation='relu')
    convnet = max_pool_2d(convnet, kernel_size=2, strides=1, padding='valid')

    convnet = flatten(convnet)

    convnet = fully_connected(convnet, n_units=1800, weights_init='truncated_normal', activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, n_units=900, weights_init='truncated_normal', activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, n_units=output_size, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    return model
    
'''


#CONFIGURATION 3
'''
def create2dConvNetNeuralNetworkModel(input_size, output_size, learningRate):

    convnet = input_data(shape=[None, input_size, input_size,1], name='input')

    #tflearn.init_graph(num_cores=1, gpu_memory_fraction=0.5)

    convnet = conv_2d(convnet, nb_filter=10, filter_size=5, strides=1, padding='same', activation='relu')
    convnet = max_pool_2d(convnet, kernel_size=2, strides=1, padding='valid')

    convnet = conv_2d(convnet, nb_filter=20, filter_size=3, strides=1, padding='valid', activation='relu')
    convnet = max_pool_2d(convnet, kernel_size=2, strides=1, padding='valid')

    convnet = flatten(convnet)

    convnet = fully_connected(convnet, n_units=3000, weights_init='truncated_normal', activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, n_units=3000, weights_init='truncated_normal', activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, n_units=output_size, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    return model
'''

# CONFIGURATION 4
# configuration is too large to be processed!
'''
def create2dConvNetNeuralNetworkModel(input_size, output_size, learningRate):

    convnet = input_data(shape=[None, input_size, input_size,1], name='input')

    #tflearn.init_graph(num_cores=1, gpu_memory_fraction=0.7)

    convnet = conv_2d(convnet, nb_filter=10, filter_size=7, strides=1, padding='same', activation='relu')
    convnet = max_pool_2d(convnet, kernel_size=2, strides=1, padding='valid')

    convnet = conv_2d(convnet, nb_filter=20, filter_size=5, strides=1, padding='valid', activation='relu')
    convnet = max_pool_2d(convnet, kernel_size=2, strides=1, padding='valid')

    convnet = conv_2d(convnet, nb_filter=30, filter_size=3, strides=1, padding='valid', activation='relu')
    convnet = max_pool_2d(convnet, kernel_size=2, strides=1, padding='valid')

    convnet = flatten(convnet)

    convnet = fully_connected(convnet, n_units=3000, weights_init='truncated_normal', activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, n_units=1500, weights_init='truncated_normal', activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, n_units=output_size, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    return model

'''

#CONFIGURATION 5

def create2dConvNetNeuralNetworkModel(input_size, output_size, learningRate):

    # Specify the log directory
    logdir = 'log/2d_combined/' + datetime.now().strftime("%Y%m%d-%H%M%S")

    convnet = input_data(shape=[None, input_size, input_size,1], name='input')

    #tflearn.init_graph(num_cores=1, gpu_memory_fraction=0.7)

    convnet = conv_2d(convnet, nb_filter=5, filter_size=9, strides=1, padding='same', activation='relu')
    convnet = max_pool_2d(convnet, kernel_size=2, strides=1, padding='valid')

    convnet = conv_2d(convnet, nb_filter=10, filter_size=7, strides=1, padding='valid', activation='relu')
    convnet = max_pool_2d(convnet, kernel_size=2, strides=1, padding='valid')

    convnet = conv_2d(convnet, nb_filter=15, filter_size=5, strides=1, padding='valid', activation='relu')
    convnet = max_pool_2d(convnet, kernel_size=2, strides=1, padding='valid')

    convnet = conv_2d(convnet, nb_filter=20, filter_size=3, strides=1, padding='valid', activation='relu')
    convnet = max_pool_2d(convnet, kernel_size=2, strides=1, padding='valid')

    convnet = flatten(convnet)

    convnet = fully_connected(convnet, n_units=1800, weights_init='truncated_normal', activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, n_units=900, weights_init='truncated_normal', activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, n_units=450, weights_init='truncated_normal', activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, n_units=output_size, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    return model
