import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


# TODO: Define/tune the layers
def create2dConvNetNeuralNetworkModel(input_size, output_size, learningRate):
    # TODO: Check padding
    # TODO: Check regularization
    # TODO: Check activation function
    # TODO: Check learning rate

    convnet = input_data(shape=[None, input_size, input_size,1], name='input')
    # convnet = tflearn.embedding(convnet, input_dim=input_size, output_dim=3)

    tflearn.init_graph(num_cores=1, gpu_memory_fraction=0.5)

    # convolutionalNetwork = conv_1d(convolutionalNetwork, 2, 1, activation='relu')
    convnet = conv_2d(convnet, 32, 2, activation='relu')
    # convolutionalNetwork = max_pool_1d(convolutionalNetwork, 2)
    convnet = max_pool_2d(convnet, 2)

    # convolutionalNetwork = conv_1d(convolutionalNetwork, 4, 1, activation='relu')
    convnet = conv_2d(convnet, 64, 2, activation='relu')
    # convolutionalNetwork = max_pool_1d(convolutionalNetwork, 2)
    convnet = max_pool_2d(convnet, 2)

    #TODO: Flatten?

    # convolutionalNetwork = fully_connected(convolutionalNetwork, 4 , activation='relu')
    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, output_size, activation='softmax')
    # Regression layer is used for backpropagation
    convnet = regression(convnet, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy',
                         name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir="log")

    return model