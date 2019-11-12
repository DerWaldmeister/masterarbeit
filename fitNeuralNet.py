from convnet_1d import create1dConvNetNeuralNetworkModel
from convnet_2d import create2dConvNetNeuralNetworkModel
from combined_convnet_2d import createCombined2dConvNetNeuralNetworkModelForFutureResourceUtilisation
from combined_convnet_1d import createCombined1dConvNetNeuralNetworkModelForFutureResourceUtilisation

def fit1DimensionalConvnet(neuralNetworkType, learningRate, numberOfEpochs, states, actions, statesValidationSet, actionsValidationSet):
    # 1dimensional convnet without using futureResoureUtilisationMatrix
    if neuralNetworkType == "1dimensional convnet":

        #if importExistingNeuralNetworkModel:
        #    print("check if a neural network model exists")
        #    if neuralNetworkModelAlreadyExists:
        #        print("import neural network model exists")
        #
        #    else:
        #        neuralNetworkModel = create1dConvNetNeuralNetworkModel(len(states[0]), len(actions[0]), learningRate)
        #        # neuralNetworkModel = createNeuralNetworkModel(len(states[0]), len(actionsPossibilities[0]), learningRate)
        #else:
        #    neuralNetworkModel = create1dConvNetNeuralNetworkModel(len(states[0]), len(actions[0]), learningRate)
        #    # neuralNetworkModel = createNeuralNetworkModel(len(states[0]), len(actionsPossibilities[0]), learningRate)
        neuralNetworkModel = create1dConvNetNeuralNetworkModel(len(states[0]), len(actions[0]), learningRate)

        runId = "1d_config_1_lr" + str(learningRate) + "_epochs" + str(numberOfEpochs)

        neuralNetworkModel.fit({"input_currentState": states}, {"targets": actions}, n_epoch=numberOfEpochs, snapshot_epoch=True, validation_set=(statesValidationSet, actionsValidationSet),
                               show_metric=True, run_id=runId)

    return neuralNetworkModel, runId