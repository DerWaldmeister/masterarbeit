# Imports
import time
import os
import glob
import random
import xlsxwriter
import re
import tflearn
import statistics as st
import numpy as np
from tflearn.layers.conv import conv_1d, conv_2d, max_pool_1d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from itertools import chain, combinations  # generation of powersets


def runSimulation(runSimulationInput):

    currentIndexActivitySequence = runSimulationInput.indexActivitySequence
    numberOfSimulationRuns = runSimulationInput.numberOfSimulationRuns
    timeDistribution = runSimulationInput.timeDistribution
    purpose = runSimulationInput.purpose
    randomDecisionProbability = runSimulationInput.randomDecisionProbability
    policyType = runSimulationInput.policyType
    decisionTool = runSimulationInput.decisionTool

    currentActivitySequence = activitySequences[currentIndexActivitySequence]

    print("start " + str(currentActivitySequence.fileName[:-4]))

    # reset variables for the series of runs
    indexSimulationRun = 0
    # reset lists for the series of runs
    totalDurations = []
    luckFactors = []
    trivialDecisionPercentages = []
    stateActionPairsOfRuns = []
    while indexSimulationRun < numberOfSimulationRuns:
        # reset variables for run
        sumActivityDuration = 0
        step = 0
        numberOfDecisions = 0
        numberOfTrivialDecisions = 0
        # reset lists for run
        if purpose == "generateData":
            currentStateActionPairsOfRun = []
        # reset dynamic variables of classes for run
        currentActivitySequence.availableResources = currentActivitySequence.totalResources[:]
        currentActivitySequence.virtualTime = 0
        # Reset variables of all activities of the currentActivitySequence
        for i in range(len(currentActivitySequence.activities)):
            currentActivitySequence.activities[i].withToken = False
            currentActivitySequence.activities[i].idleToken = False
            currentActivitySequence.activities[i].numberOfCompletedPreviousActivities = 0
            currentActivitySequence.activities[i].remainingTime = 0
            currentActivitySequence.activities[i].processedTime = 0
            currentActivitySequence.activities[i].seizedResources = [0] * numberOfResources

        # set startActivities ready to start
        for indexStartActivity in currentActivitySequence.indexStartActivities:
            currentActivitySequence.activities[indexStartActivity].withToken = True
            currentActivitySequence.activities[indexStartActivity].idleToken = True

        # start simulation
        simulationRunFinished = False
        while simulationRunFinished == False:    # if there are some token left in some activities
            step += 1

            ## STEP 1 ##
            # 1.1 find activities ready to start and store in list indexReadyToStartActivities
            indexReadyToStartActivities = []
            for i, currentActivity in enumerate(currentActivitySequence.activities):
                if (currentActivity.withToken and currentActivity.idleToken and currentActivity.numberOfCompletedPreviousActivities == currentActivity.numberOfPreviousActivities):
                    # verify that enough resources are available to start
                    enoughResourcesAreAvailable = True
                    for j in range(numberOfResources):
                        if currentActivity.requiredResources[j] > currentActivitySequence.availableResources[j]:
                            enoughResourcesAreAvailable = False
                            break
                    if enoughResourcesAreAvailable:
                        # Activity is ready to start
                        indexReadyToStartActivities.append(i)

            # 1.2 check if the decision is trivial
            trivialDecision = True
            # Compute powerset of indexReadyToStartActivities and create a list of the powerset
            indexReadyToStartActivitiesPowerset = list(powerset(indexReadyToStartActivities))
            # find feasible combined decisions_indexActivity (only resource check needed) (Check which subset of activities can be executed together?)
            feasibleCombinedDecisions_indexActivity = []
            #  Reversed: Start with biggest subset
            for i in reversed(range(len(indexReadyToStartActivitiesPowerset))):
                currentDecision = list(indexReadyToStartActivitiesPowerset[i])
                decisionIsASubsetOfFeasibleDecision = False
                for j,feasibleDecisionAlreadyInList in enumerate(feasibleCombinedDecisions_indexActivity):
                    if len(set(currentDecision) - set(feasibleDecisionAlreadyInList)) == 0:
                        decisionIsASubsetOfFeasibleDecision = True
                        break
                if decisionIsASubsetOfFeasibleDecision == False:
                    # verify that enough resources are available to start all the activities
                    totalRequiredResources = [0] * numberOfResources
                    for indexCurrentActivity in currentDecision:
                        for j in range(numberOfResources):
                            totalRequiredResources[j] += currentActivitySequence.activities[indexCurrentActivity].requiredResources[j]
                    enoughResourcesAreAvailable = True
                    for j in range(numberOfResources):
                        if totalRequiredResources[j] > currentActivitySequence.availableResources[j]:
                            enoughResourcesAreAvailable = False
                            break
                    if enoughResourcesAreAvailable:
                        feasibleCombinedDecisions_indexActivity.append(currentDecision)
            if len(feasibleCombinedDecisions_indexActivity) > 1:
                trivialDecision = False

            numberOfDecisions += 1
            if trivialDecision:
                numberOfTrivialDecisions +=1

            # 1.3 define activity conversion vector and resource conversion vector
            # initialise activityConversionVector and ResourceConversionVector
            activityConversionVector = [-1] * numberOfActivitiesInStateVector
            activityScores = []
            indexReadyToStartActivitiesInState = indexReadyToStartActivities[0:min(numberOfActivitiesInStateVector, len(indexReadyToStartActivities))]
            if trivialDecision:
                # no conversion needed
                resourceConversionVector = list(range(0,numberOfResources))
                for i in range(len(indexReadyToStartActivitiesInState)):
                    activityConversionVector[i] = indexReadyToStartActivitiesInState[i]
            else:
                # conversion is required
                # find most critical resources (i.e. required resources to start the ready to start activities normalized by the total resources)
                resourceNeedForReadyToStartActivities = [0] * numberOfResources
                for i in indexReadyToStartActivities:
                    for j in range(numberOfResources):
                        resourceNeedForReadyToStartActivities[j] += currentActivitySequence.activities[i].requiredResources[j] / currentActivitySequence.totalResources[j]
                # create resourceConversionVector
                indexResourcesGlobal = list(range(0,numberOfResources))
                # Sort indexResourcesGlobal based on resourceNeedForReadyToStartActivities
                indexResourcesGlobal_reordered = [x for _, x in sorted(zip(resourceNeedForReadyToStartActivities, indexResourcesGlobal), reverse=True)]
                resourceConversionVector = indexResourcesGlobal_reordered
                # reorder activities depending on resource utilisation
                activityScores = [-1] * numberOfActivitiesInStateVector
                for i in range(len(indexReadyToStartActivitiesInState)):
                    for j in range(len(resourceConversionVector)):
                        resourceMultiplicator = 100 ** (numberOfResources-j-1)
                        resourceQuantity = currentActivitySequence.activities[indexReadyToStartActivitiesInState[i]].requiredResources[resourceConversionVector[j]]
                        activityScores[i] += 1 + resourceMultiplicator * resourceQuantity

                indexActivitiesGlobal = [-1] * numberOfActivitiesInStateVector
                indexActivitiesGlobal[0:len(indexReadyToStartActivitiesInState)] = indexReadyToStartActivitiesInState
                # Sort indexActivitiesGlobal based on activityScores
                indexActivitiesGlobal_reordered = [x for _, x in sorted(zip(activityScores, indexActivitiesGlobal), reverse=True)]
                activityConversionVector = indexActivitiesGlobal_reordered

            # 1.4 normalized state vector and matrix are created
            # This line : currentState_readyToStartActivities = [] changed to:
            currentState_readyToStartActivitiesMatrix = [[]]
            if trivialDecision == False:
                currentState_readyToStartActivities = np.zeros((1, stateVectorLength))
                for i, indexActivity in enumerate(activityConversionVector):
                    if indexActivity != -1:
                        currentState_readyToStartActivities[0, 0+i*(1+numberOfResources)] = currentActivitySequence.activities[indexActivity].time * rescaleFactorTime
                        for j in range(numberOfResources):
                            currentState_readyToStartActivities[0, 1 + j + i * (1 + numberOfResources)] = currentActivitySequence.activities[indexActivity].requiredResources[resourceConversionVector[j]] / currentActivitySequence.totalResources[resourceConversionVector[j]]
                for j in range(numberOfResources):
                    currentState_readyToStartActivities[0, numberOfActivitiesInStateVector + numberOfActivitiesInStateVector * numberOfResources + j] = currentActivitySequence.availableResources[resourceConversionVector[j]] / currentActivitySequence.totalResources[resourceConversionVector[j]]
            # (optional: add information about the future resource utilisation)
            # determine the earliest starting point of each activity considering the problem without resource constraints and deterministic
            # currentState_futureResourceUtilisation = np.zeros([numberOfResources, timeHorizon])

                # Multiply the currentState_readyToStartActivities vector with its transpose vector in order to get a matrix
                # currentState_readyToStartActivities is in theformat of (1, stateVectorLength)
                currentState_readyToStartActivitiesVertical = currentState_readyToStartActivities.reshape(stateVectorLength, 1)
                #print("currentState_readyToStartActivities: " + str(currentState_readyToStartActivities))
                #print("currentState_readyToStartActivitiesVertical: " + str(currentState_readyToStartActivitiesVertical))
                currentState_readyToStartActivitiesMatrix = np.matmul(currentState_readyToStartActivitiesVertical, currentState_readyToStartActivities)
                #print("currentState_readyToStartActivitiesMatrix: " + str(currentState_readyToStartActivitiesMatrix))



            # 1.5 Use the policy and the decision tool to define which tokens can begin the correspondent activity or remain idle
            # Random decision at this step?
            randomDecisionAtThisStep = (random.random() < randomDecisionProbability)
            if trivialDecision:    # if the decision is trivial, it does not matter how the priority values are assigned
                randomDecisionAtThisStep = True
            if randomDecisionAtThisStep:
                priorityValues = np.random.rand(numberOfActivitiesInStateVector)
            # No random decision
            else:
                if policyType == "neuralNetworkModel":
                    # Disabled by me: currentState_readyToStartActivities = currentState_readyToStartActivities.reshape(-1, stateVectorLength)
                    # Reshape the currentState_readyToStartActivitiesMatrix so that it fits into input shape of the neural network
                    currentState_readyToStartActivitiesMatrix = currentState_readyToStartActivitiesMatrix.reshape([-1,stateVectorLength,stateVectorLength,1])
                    outputNeuralNetworkModel = decisionTool.predict(currentState_readyToStartActivitiesMatrix)
                    # print("outputNeuralNetworkModel: " + str(outputNeuralNetworkModel))
                    # outputNeuralNetworkModel: [[0.26827973 0.32891855 0.26756716 0.06040468 0.03964273 0.03518711]]
                    # outputNeuralNetworkModel: [[0.26801276 0.32818934 0.2673128  0.06086484 0.04005156 0.03556869]]
                    priorityValues = np.zeros(numberOfActivitiesInStateVector)
                    for i in range(len(outputNeuralNetworkModel)):
                        priorityValues[i] = outputNeuralNetworkModel[0,i]
                    # print("priorityValues: " + str(priorityValues) )
                    # priorityValues: [0.26827973 0.         0.         0.         0.         0.        ]
                    # priorityValues: [0.26801276 0.         0.         0.         0.         0.        ]

                elif policyType == "otherPolicy1":
                    print("generate priority values with other policy 1")
                elif policyType == "otherPolicy2":
                    print("generate priority values with other policy 2")
                else:
                    print("policy name not existing")

            # reorder list according to priority
            decisions_indexActivity_reordered = [x for _, x in sorted(zip(priorityValues,decisions_indexActivity), reverse=True)]

            # use the priority values to start new activities
            currentAction = np.zeros([numberOfActivitiesInStateVector])
            indexStartedActivities = []
            # consider the decision one by one in reordered list
            for indexActivityToStartLocal in decisions_indexActivity_reordered:
                indexActivityToStartGlobal = activityConversionVector[indexActivityToStartLocal]
                if indexActivityToStartGlobal != -1:
                    currentActivity = currentActivitySequence.activities[indexActivityToStartGlobal]
                    if currentActivity.withToken and currentActivity.idleToken and currentActivity.numberOfCompletedPreviousActivities == currentActivity.numberOfPreviousActivities:
                        # verify that enough resources are available to start
                        enoughResourcesAreAvailable = True
                        for i in range(numberOfResources):
                            if currentActivity.requiredResources[i] > currentActivitySequence.availableResources[i]:
                                enoughResourcesAreAvailable = False
                                break
                        if enoughResourcesAreAvailable:
                            currentActivitySequence.activities[indexActivityToStartGlobal].idleToken = False

                            # 1.6 Set remaining time for the starting activity
                            if timeDistribution == "deterministic":
                                currentActivitySequence.activities[indexActivityToStartGlobal].remainingTime = currentActivitySequence.activities[indexActivityToStartGlobal].time
                                sumActivityDuration += currentActivitySequence.activities[indexActivityToStartGlobal].remainingTime

                            # 1.7 seize resources
                            for i in range(numberOfResources):
                                currentActivitySequence.activities[indexActivityToStartGlobal].seizedResources[i] = currentActivitySequence.activities[indexActivityToStartGlobal].requiredResources[i]
                                currentActivitySequence.availableResources[i] -= currentActivitySequence.activities[indexActivityToStartGlobal].requiredResources[i]

                            # update the action vector with the activity that has been just started
                            currentAction[indexActivityToStartLocal] = 1
                            indexStartedActivities.append(indexActivityToStartGlobal)

            # 1.8 if the purpose is to generate training data, save the current state action pair, but not for trivial decisions
            if purpose == "generateData" and trivialDecision == False:
                currentStateActionPair = StateActionPair()
                currentStateActionPair.state = currentState_readyToStartActivitiesMatrix
                currentStateActionPair.action = currentAction
                currentStateActionPairsOfRun.append(currentStateActionPair)


            ## STEP 2 ##
            # 2.1 find out when the next event (activity end) occurs
            smallestRemainingTime = 1e300
            indexActiveActivities = []
            for i in range(numberOfActivities):
                if currentActivitySequence.activities[i].withToken and currentActivitySequence.activities[i].idleToken == False:
                    indexActiveActivities.append(i)
                    # Iterate over all activities and set smallestRemainingTime as the smallest remainingTime of all activities
                    if currentActivitySequence.activities[i].remainingTime < smallestRemainingTime:
                        smallestRemainingTime = currentActivitySequence.activities[i].remainingTime
            # 2.2 find next finishing activities
            # indexNextFinishingActivities are those activities that have as remainingTime the smallestRemainingTime
            indexNextFinishingActivities = []
            for i in indexActiveActivities:
                if currentActivitySequence.activities[i].remainingTime == smallestRemainingTime:
                    indexNextFinishingActivities.append(i)
            # 2.3 jump forward to activity end
            # Update remainingTime and processedTime
            currentActivitySequence.virtualTime += smallestRemainingTime
            for i in indexActiveActivities:
                currentActivitySequence.activities[i].remainingTime -= smallestRemainingTime
                currentActivitySequence.activities[i].processedTime += smallestRemainingTime

            ## STEP 3 ##
            # for each just finished activity:
            for i in indexNextFinishingActivities:
                # 3.1 find following activities
                indexFollowingActivities = currentActivitySequence.activities[i].indexFollowingActivities
                # 3.2 for each following activity, increment the numberOfCompletedPreviousActivities and, if there is no token already in the following activity, add an idle token.
                for j in indexFollowingActivities:
                    currentActivitySequence.activities[j].numberOfCompletedPreviousActivities += 1
                    if currentActivitySequence.activities[j].withToken == False:
                        currentActivitySequence.activities[j].withToken = True
                        currentActivitySequence.activities[j].idleToken = True
                # 3.3 delete token from just finished activity
                currentActivitySequence.activities[i].withToken = False
                currentActivitySequence.activities[i].idleToken = False
                # 3.4 release resources back to the resource pool
                currentActivitySequence.activities[i].seizedResources = [0] * numberOfResources
                for j in range(numberOfResources):
                    currentActivitySequence.availableResources[j] += currentActivitySequence.activities[i].requiredResources[j]

            ## STEP 4 ##
            # check if all activities are completed (i.e. no more token)
            simulationRunFinished = True
            for i in range(numberOfActivities):
                if currentActivitySequence.activities[i].withToken:
                    simulationRunFinished = False
                    break

            #if step%10 == 0:
            #    print("step:" + str(step))

        totalDuration = currentActivitySequence.virtualTime
        luckFactor = sumActivityDuration / sum(a.time for a in currentActivitySequence.activities)
        trivialDecisionPercentage = numberOfTrivialDecisions / numberOfDecisions

        totalDurations.append(totalDuration)
        luckFactors.append(luckFactor)
        trivialDecisionPercentages.append(trivialDecisionPercentage)

        if purpose == "generateData":
            stateActionPairsOfRuns.append(currentStateActionPairsOfRun)

        # increment the index for the simulation run at the end of the loop
        indexSimulationRun += 1
        # if indexSimulationRun % 10 == 0:
            # print("indexSimulationRun:" + str(step))


    totalDurationMean = st.mean(totalDurations)
    totalDurationStDev = None
    if numberOfSimulationRuns != 1:
        totalDurationStDev = st.stdev(totalDurations)
    totalDurationMin = min(totalDurations)
    totalDurationMax = max(totalDurations)
    luckFactorMean = st.mean(luckFactors)
    trivialDecisionPercentageMean = st.mean(trivialDecisionPercentages)

    currentRunSimulationOutput = RunSimulationOutput()
    currentRunSimulationOutput.totalDurationMean = totalDurationMean
    currentRunSimulationOutput.totalDurationStDev = totalDurationStDev
    currentRunSimulationOutput.totalDurationMin = totalDurationMin
    currentRunSimulationOutput.totalDurationMax = totalDurationMax
    currentRunSimulationOutput.luckFactorMean = luckFactorMean
    currentRunSimulationOutput.trivialDecisionPercentageMean = trivialDecisionPercentageMean

    # submit the stateActionPairs of the best run, if the standard deviation of the duration is not zero
    if purpose == "generateData":
        if totalDurationStDev != 0:
            # TODO: Warum ist indexBestRun nicht bei totalDurationMin ?
            indexBestRun = totalDurations.index(totalDurationMax)
            currentRunSimulationOutput.stateActionPairsOfBestRun = stateActionPairsOfRuns[indexBestRun]

    print("end " + str(currentActivitySequence.fileName[:-4]))

    return currentRunSimulationOutput


# TODO: Define/tune the layers
def createNeuralNetworkModel(input_size, output_size):
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


# return powerset (Potenzmenge) of a set "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
def powerset(listOfElements):
    s = list(listOfElements)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

##################################################    CLASSES   ##################################################

# Class to represent a topology
class ActivitySequence:
    def __init__(self):
        # static (do not change during simulation)
        self.index = None
        self.fileName = None
        self.numberOfResources = None
        self.totalResources = []    # Capacities of the resources of an activitySequence
        self.numberOfActivities = None
        self.activities = []
        self.indexStartActivities = []
        # dynamic (change during simulation)
        self.availableResources = []
        self.totalDurationMean = 0
        self.totalDurationStandardDeviation = None
        self.totalDurationMin = None
        self.totalDurationMax = None
        self.luckFactorMean = None
        self.totalDurationWithPolicy = None
        self.trivialDecisionPercentageMean = None


# Class to represent activity of topologies
class Activity:
    def __init__(self):
        # static (do not change during simulation)
        self.index = None
        self.time = None  # expected value. Only deterministic component. The distribution is given as an argument in the function run simulation.
        self.requiredResources = []
        self.numberOfPreviousActivities = 0
        self.indexFollowingActivities = []
        # dynamic (change during simulation)
        self.withToken = None
        self.idleToken = None
        self.numberOfCompletedPreviousActivities = None
        self.remainingTime = None  # time to activity end
        self.processedTime = None  # time elapsed from the beginning of the activity
        self.seizedResources = []


# Input parameters for the simulation
class RunSimulationInput:
    def __init__(self):
        self.indexActivitySequence = None
        self.numberOfSimulationRuns = None
        self.timeDistribution = None
        self.purpose = None     # "generateData", "testPolicy"
        self.randomDecisionProbability = None   # 1 or 0: 1 = all decisions randomly, 0 = no random decisions
        self.policyType = None  # None or NeuralNetworkModel
        self.decisionTool = None    # None or NeuralNetworkModel


# Output parameters of the simulation
class RunSimulationOutput:
    def __init__(self):
        self.totalDurationMean = None
        self.totalDurationStDev = None
        self.totalDurationMin = None
        self.totalDurationMax = None
        self.luckFactorMean = None
        self.trivialDecisionPercentageMean = None
        self.stateActionPairsOfBestRun = []


class StateActionPair:
    def __init__(self):
        self.state = None
        self.action = None



##################################################    MAIN    ##################################################
t_start = time.time()
# user defined parameters
# problem parameters
timeDistribution = "deterministic"  # deterministic, exponential, uniform_1, uniform_2, ...
# CPU parameters
numberOfCpuProcessesToGenerateData = 1  # paoloPC has 16 cores
maxTasksPerChildToGenerateData = 1  # 4 is the best for paoloPC
# input state vector  parameters
numberOfActivitiesInStateVector = 6
rescaleFactorTime = 0.1
timeHorizon = 10
# random generation parameters
numberOfSimulationRunsToGenerateData = 1000
numberOfSimulationRunsToTestPolicy = 1
# train parameters
percentageOfFilesTest = 0.1
importExistingNeuralNetworkModel = False
numberOfEpochs = 3
learningRate = 0.001
# paths
relativePath = os.path.dirname(__file__)
# Path for PSPLIB J30
absolutePathProjects = relativePath + "/database/psplib J30"
# Path for random RanGen data from Xialoei
# absolutePathProjects = relativePath + "/database/RG30_Newdata"
absolutePathExcelOutput = relativePath + "/BenchmarkCNN2d.xlsx"
# initialise variables
numberOfActivities = None
numberOfResources = None
activitySequences = []
decisions_indexActivity = []
decisions_indexActivitiyPowerset = []
states = []
actions = []

# read all activity sequences from database
absolutePathProjectsGlob = absolutePathProjects + "*.txt"
# Generates a list containing all the files in the directory
files = sorted(glob.glob(absolutePathProjectsGlob))
# print("v: files: ", files)

# divide all activity sequences in training and test set
numberOfFiles = len(files)
# print("V: numberOfFiles: ", numberOfFiles)
numberOfFilesTest = round(numberOfFiles * percentageOfFilesTest)
# print(numberOfFilesTest)
numberOfFilesTrain = numberOfFiles - numberOfFilesTest
indexFiles = list(range(0, numberOfFiles))
# print("v: indexFiles (topologies get indexed):", indexFiles)
indexFilesTrain = []
indexFilesTest = []
# randomly choose test files
for i in range(numberOfFilesTest):
    randomIndex = random.randrange(0, len(indexFiles))
    indexFilesTest.append(indexFiles[randomIndex])
    del indexFiles[randomIndex]
indexFilesTrain = indexFiles
# print("v: indexFilesTrain:", indexFilesTrain)
# print("v: indexFilesTest:", indexFilesTest)

# organize the read activity sequences in classes
for i in range(numberOfFiles):
    file = files[i]
    # print(File)
    # create a new activitySequence object
    currentActivitySequence = ActivitySequence()
    # print(currentActivitySequence)
    # Opens file and reads it
    with open(file, "r") as f:
        currentActivitySequence.index = i
        currentActivitySequence.fileName = os.path.basename(f.name)
        # print(currentActivitySequence.fileName)
        # allLines = f.read()
        # print(allLines)
        firstLine = f.readline()  # information about numberOfActivities and numberOfResourceTypes
        firstLineDecomposed = re.split(" +", firstLine)
        numberOfActivities = (int(firstLineDecomposed[0]) - 2)  # the first and last dummy activity do not count
        currentActivitySequence.numberOfActivities = numberOfActivities
        # print("numberOfActivities = " + str(currentActivitySequence.numberOfActivities))
        secondLine = f.readline()  # information about total available resources
        secondLineDecomposed = re.split(" +", secondLine)
        # print("Available amount per resource:", secondLineDecomposed)
        numberOfResources = 0
        for totalResources in secondLineDecomposed[0:-1]:
            numberOfResources += 1
            currentActivitySequence.totalResources.append(int(totalResources))
        currentActivitySequence.numberOfResources = numberOfResources
        # print("numberOfResources per activity sequence: ", currentActivitySequence.numberOfResources)
        thirdLine = f.readline()  # information about starting activities
        thirdLineDecomposed = re.split(" +", thirdLine)
        # print("Starting acitivites unformatted: ", thirdLineDecomposed)
        for idActivity in thirdLineDecomposed[6:-1]:
            currentActivitySequence.indexStartActivities.append(int(idActivity) - 2)
        # print("indexStartActivities = " + str(currentActivitySequence.indexStartActivities))
        line = f.readline()
        while line:
            # print(line, end="")
            lineDecomposed = re.split(" +", line)
            if int(lineDecomposed[0]) == 0:
                break
            else:
                currentActivity = Activity()
                currentActivity.time = int(lineDecomposed[0])
                currentActivity.requiredResources = [int(lineDecomposed[1]), int(lineDecomposed[2]),
                                                     int(lineDecomposed[3]), int(lineDecomposed[4])]
                for idFollowingActivity in lineDecomposed[6:-1]:
                    if int(
                            idFollowingActivity) != numberOfActivities + 2:  # if the following action is not the last dummy activity
                        currentActivity.indexFollowingActivities.append(int(idFollowingActivity) - 2)
            currentActivitySequence.activities.append(currentActivity)
            line = f.readline()
            # add indexes to list of activities
        for j in range(len(currentActivitySequence.activities)):
            currentActivitySequence.activities[j].index = j

        # add numberOfPreviousActivities
        for activity in currentActivitySequence.activities:
            for indexFollowingActivity in activity.indexFollowingActivities:
                currentActivitySequence.activities[indexFollowingActivity].numberOfPreviousActivities += 1
    activitySequences.append(currentActivitySequence)



# Formel fuer stateVectorLength?
stateVectorLength = numberOfActivitiesInStateVector + numberOfActivitiesInStateVector * numberOfResources + numberOfResources
# print("stateVectorLength:", stateVectorLength)

# compute decisions: each decision corresponds to a start of an activity in the local reference system
# (more than one decision can be taken at once)
for i in range(0, numberOfActivitiesInStateVector):
    decisions_indexActivity.append(i)
    # print(decisions_indexActivity)



#####################__SIMULATION_RUNS__#################

####  GENERATE TRAINING DATA USING RANDOM DECISIONS (WITHOUT USING pool.map) ####
print("Start generating training data using random decisions...")

for i in range(numberOfFilesTrain):
    # currentRunSimulation_input is an object of RunSimulationInput()
    currentRunSimulationInput = RunSimulationInput()
    currentRunSimulationInput.indexActivitySequence = indexFilesTrain[i]
    currentRunSimulationInput.numberOfSimulationRuns = numberOfSimulationRunsToGenerateData
    currentRunSimulationInput.timeDistribution = timeDistribution
    currentRunSimulationInput.purpose = "generateData"
    currentRunSimulationInput.randomDecisionProbability = 1
    currentRunSimulationInput.policyType = None
    currentRunSimulationInput.decisionTool = None
    currentRunSimulationOutput = runSimulation(currentRunSimulationInput)
    activitySequences[indexFilesTrain[i]].totalDurationMean = currentRunSimulationOutput.totalDurationMean
    activitySequences[
        indexFilesTrain[i]].totalDurationStandardDeviation = currentRunSimulationOutput.totalDurationStDev
    activitySequences[indexFilesTrain[i]].totalDurationMin = currentRunSimulationOutput.totalDurationMin
    activitySequences[indexFilesTrain[i]].totalDurationMax = currentRunSimulationOutput.totalDurationMax
    activitySequences[indexFilesTrain[i]].luckFactorMean = currentRunSimulationOutput.luckFactorMean
    activitySequences[
        indexFilesTrain[i]].trivialDecisionPercentageMean = currentRunSimulationOutput.trivialDecisionPercentageMean
    for currentStateActionPair in currentRunSimulationOutput.stateActionPairsOfBestRun:
        states.append(currentStateActionPair.state)
        actions.append(currentStateActionPair.action)

print("states[0]: " + str(states[0]))
#print("states[1]: " + str(states[1]))
#print("len(states[0]: " + str(len(states[0])))
#print("(len(states[0]),len(states[0])): " + str((len(states[0]),len(states[0]))))

# Turn states list into tuples
states = np.asarray(states)
print("states[0]: " + str(states[0]))
# Reshape states
states = states.reshape([-1,len(states[0]),len(states[0]),1])

###  TRAIN MODEL USING TRAINING DATA  ####
# Look for existing model
print("Train model using training data...")
if importExistingNeuralNetworkModel:
    neuralNetworkModelAlreadyExists = False
    print("check if a neural network model exists")
    if neuralNetworkModelAlreadyExists:
        print("import neural network model exists")
    else:
        neuralNetworkModel = createNeuralNetworkModel(len(states[0]), len(actions[0]))
else:
    neuralNetworkModel = createNeuralNetworkModel(len(states[0]), len(actions[0]))

neuralNetworkModel.fit({"input": states}, {"targets": actions}, n_epoch=numberOfEpochs, snapshot_epoch=500,
                       show_metric=True, run_id="trainNeuralNetworkModel")

####  CREATE BENCHMARK WITH RANDOM DECISIONS ALSO WITH TEST ACTIVITY SEQUENCES  ####
print("Create benchmark on test activity sequences using random decisions...")
for i in range(numberOfFilesTest):
    currentRunSimulationInput = RunSimulationInput()
    currentRunSimulationInput.indexActivitySequence = indexFilesTest[i]
    currentRunSimulationInput.numberOfSimulationRuns = numberOfSimulationRunsToGenerateData
    currentRunSimulationInput.timeDistribution = timeDistribution
    currentRunSimulationInput.purpose = "testPolicy"
    currentRunSimulationInput.randomDecisionProbability = 1
    currentRunSimulationInput.policyType = None
    currentRunSimulationInput.decisionTool = None
    currentRunSimulationOutput = runSimulation(currentRunSimulationInput)
    activitySequences[indexFilesTest[i]].totalDurationMean = currentRunSimulationOutput.totalDurationMean
    activitySequences[indexFilesTest[i]].totalDurationStandardDeviation = currentRunSimulationOutput.totalDurationStDev
    activitySequences[indexFilesTest[i]].totalDurationMin = currentRunSimulationOutput.totalDurationMin
    activitySequences[indexFilesTest[i]].totalDurationMax = currentRunSimulationOutput.totalDurationMax
    activitySequences[indexFilesTest[i]].luckFactorMean = currentRunSimulationOutput.luckFactorMean
    activitySequences[indexFilesTest[i]].trivialDecisionPercentageMean = currentRunSimulationOutput.trivialDecisionPercentageMean

####  TEST NEURAL NETWORK MODEL ON TRAIN ACTIVITY SEQUENCES  ####
# run simulations with neural network model as decision tool
# (not possible to use multiprocessing -> apparently is not possible to parallelize processes on GPU)
for i in range(numberOfFilesTrain):
    currentRunSimulationInput = RunSimulationInput()
    currentRunSimulationInput.indexActivitySequence = indexFilesTrain[i]
    currentRunSimulationInput.numberOfSimulationRuns = numberOfSimulationRunsToTestPolicy
    currentRunSimulationInput.timeDistribution = timeDistribution
    currentRunSimulationInput.purpose = "testPolicy"
    currentRunSimulationInput.randomDecisionProbability = 0
    currentRunSimulationInput.policyType = "neuralNetworkModel"
    currentRunSimulationInput.decisionTool = neuralNetworkModel
    currentRunSimulationOutput = runSimulation(currentRunSimulationInput)
    activitySequences[indexFilesTrain[i]].totalDurationWithPolicy = currentRunSimulationOutput.totalDurationMean

####  EVALUATION OF RESULTS OF TRAIN ACTIVITY SEQUENCES  ####
sumTotalDurationRandomTrain = 0
sumTotalDurationWithNeuralNetworkModelTrain = 0
for i in range(numberOfFilesTrain):
    sumTotalDurationRandomTrain += activitySequences[indexFilesTrain[i]].totalDurationMean
    sumTotalDurationWithNeuralNetworkModelTrain += activitySequences[indexFilesTrain[i]].totalDurationWithPolicy

####  TEST NEURAL NETWORK MODEL ON TEST ACTIVITY SEQUENCES  ####
# run simulations with neural network model as decision tool
# (not possible to use multiprocessing -> apparently is not possible to parallelize processes on GPU)
print("Test neural network model on test activity sequences...")
for i in range(numberOfFilesTest):
    currentRunSimulation_input = RunSimulationInput()
    currentRunSimulation_input.indexActivitySequence = indexFilesTest[i]
    currentRunSimulation_input.numberOfSimulationRuns = numberOfSimulationRunsToTestPolicy
    currentRunSimulation_input.timeDistribution = timeDistribution
    currentRunSimulation_input.purpose = "testPolicy"
    currentRunSimulation_input.randomDecisionProbability = 0
    currentRunSimulation_input.policyType = "neuralNetworkModel"
    currentRunSimulation_input.decisionTool = neuralNetworkModel
    currentRunSimulationOutput = runSimulation(currentRunSimulation_input)
    activitySequences[indexFilesTest[i]].totalDurationWithPolicy = currentRunSimulationOutput.totalDurationMean

####  EVALUATION OF RESULTS OF TEST ACTIVITY SEQUENCES  ####
print("EVALUATION RESULTS...")
sumTotalDurationRandomTest = 0
sumTotalDurationWithNeuralNetworkModelTest = 0
for i in range(numberOfFilesTest):
    sumTotalDurationRandomTest += activitySequences[indexFilesTest[i]].totalDurationMean
    sumTotalDurationWithNeuralNetworkModelTest += activitySequences[indexFilesTest[i]].totalDurationWithPolicy

print("sumTotalDurationRandomTrain = " + str(sumTotalDurationRandomTrain))
print("sumTotalDurationWithNeuralNetworkModelTrain = " + str(sumTotalDurationWithNeuralNetworkModelTrain))
print("sumTotalDurationRandomTest = " + str(sumTotalDurationRandomTest))
print("sumTotalDurationWithNeuralNetworkModelTest = " + str(sumTotalDurationWithNeuralNetworkModelTest))

# compute computation time
t_end = time.time()
t_computation = t_end - t_start
print("t_computation = " + str(t_computation))

# write output to excel with xlsxwriter
wb = xlsxwriter.Workbook(absolutePathExcelOutput)
ws = wb.add_worksheet("J30_totalDurations")
ws.set_column("A:A", 22)
ws.write(0, 0, "number of simulation runs")
# Right number of simulation runs??
ws.write(0, 1, numberOfSimulationRunsToTestPolicy)
ws.write(1, 0, "computation time")
ws.write(1, 1, t_computation)
ws.write(2, 1, "solution random")
ws.write(3, 0, "activity sequence name")
ws.write(3, 1, "E[T]")
ws.write(3, 2, "StDev[T]")
ws.write(3, 3, "min[T]")
ws.write(3, 4, "max[T]")
ws.write(3, 5, "P[trivial decision]")
ws.write(3, 6, "min=max")
for i in indexFilesTrain:
    ws.write(i+4, 0, activitySequences[i].fileName[:-4])
    ws.write(i+4, 1, activitySequences[i].totalDurationMean)
    ws.write(i+4, 2, activitySequences[i].totalDurationStandardDeviation)
    ws.write(i+4, 3, activitySequences[i].totalDurationMin)
    ws.write(i+4, 4, activitySequences[i].totalDurationMax)
    ws.write(i+4, 5, activitySequences[i].trivialDecisionPercentageMean)
    if activitySequences[i].totalDurationMin == activitySequences[i].totalDurationMax:
        ws.write(i+4, 6, "1")
    else:
        ws.write(i+4, 6, "")
wb.close()
print("Excel workbook created")
