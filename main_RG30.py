import time
import os
import glob
import numpy as np
import random
import re
from env import runSimulation, runSimulation_input, activitySequence, activity
from convnet_1d import create1dConvNetNeuralNetworkModel
from convnet_2d import create2dConvNetNeuralNetworkModel
import multiprocessing as mp
from openpyxl import Workbook
from openpyxl.styles import Border, Alignment, Side

t_start = time.time()

# user defined parameters
# problem parameters
timeDistribution = "deterministic"    # deterministic, exponential, uniform_1, uniform_2, ...

# CPU parameters
numberOfCpuProcessesToGenerateData = 16   # paoloPC has 16 cores
maxTasksPerChildToGenerateData = 4        # 4 is the best for paoloPC


# input state vector  parameters
numberOfActivitiesInStateVector = 6
rescaleFactorTime = 0.1
timeHorizon = 10

# random generation parameters
numberOfSimulationRunsToGenerateData =2000
numberOfSimulationRunsToTestPolicy = 1

# neural network type
neuralNetworkType = "2dimensional convnet"   # 1dimensional, 2dimensional, graph embedding

# train parameters
percentageOfFilesTest = 0.1
importExistingNeuralNetworkModel = False
neuralNetworkModelAlreadyExists = False
numberOfEpochs = 50 #walk entire samples
learningRate = 0.0005

# paths
relativePath = os.path.dirname(__file__)
absolutePathProjects = relativePath + "/database/RG30_Newdata/"

# other parameters
np.set_printoptions(precision=4)    # print precision of numpy variables

# initialise variables
numberOfActivities = None
numberOfResources = None
activitySequences = []
decisions_indexActivity = []
decisions_indexActivityPowerset = []
states = []
actions = []
sumTotalDurationRandomTestRecord = []
sumTotalDurationWithNeuralNetworkModelTestRecord = []
sumTotalDurationWithCriticalResourceTestRecord = []
sumTotalDurationWithShortestProcessingTestRecord = []
sumTotalDurationWithShortestSumDurationTestRecord = []
sumTotalDurationRandomTrainRecord = []
sumTotalDurationWithNeuralNetworkModelTrainRecord = []
sumTotalDurationWithCriticalResourceTrainRecord = []
sumTotalDurationWithShortestProcessingTrainRecord = []
sumTotalDurationWithShortestSumDurationTrainRecord = []

# read all activity sequences from database
absolutePathProjectsGlob = absolutePathProjects + "*.txt"
files = sorted(glob.glob(absolutePathProjectsGlob))

# divide all activity sequences in training and test set
numberOfFiles = len(files)
numberOfFilesTest = round(numberOfFiles * percentageOfFilesTest)
numberOfFilesTrain = numberOfFiles - numberOfFilesTest
indexFiles = list(range(0, numberOfFiles))
indexFilesTrain = []
indexFilesTest = []


# choose the first element of every set to test
for i in range(numberOfFilesTest):
    # randomIndex = random.randrange(0, len(indexFiles))
    randomIndex = i*9
    indexFilesTest.append(indexFiles[randomIndex])
    del indexFiles[randomIndex]#delete
indexFilesTrain = indexFiles

# organize the read activity sequences in classes
for i in range(numberOfFiles):
    file = files[i]
    #print(file)
    # create a new activitySequence object
    currentActivitySequence = activitySequence()
    with open(file,"r") as f:
        currentActivitySequence.index = i
        currentActivitySequence.fileName = os.path.basename(f.name)
        #print(currentActivitySequence.fileName)
        # allLines = f.read()
        # print(allLines)
        next(f)
        firstLine = f.readline()    # information about numberOfActivities and numberOfResourceTypes
        firstLineDecomposed = re.split(" +", firstLine)
        numberOfActivities = (int(firstLineDecomposed[1])-2)    # the first and last dummy activity do not count
        currentActivitySequence.numberOfActivities = numberOfActivities
        # print("numberOfActivities = " + str(currentActivitySequence.numberOfActivities))
        secondLine = f.readline()   # information about total available resources
        secondLineDecomposed = re.split(" +", secondLine)
        #print(secondLineDecomposed)
        del secondLineDecomposed[0]
        secondLineDecomposed[-1]=secondLineDecomposed[-1].strip('\n')#only string have attributes of strip and delete '\n' at the end
        #print(secondLineDecomposed)
        numberOfResources = 0
        # print(len(secondLineDecomposed))
        #secondLineDecomposed=[int(secondLineDecomposed)]
        secondLineDecomposed = list(map(int,secondLineDecomposed))
        #print(secondLineDecomposed)
        for totalResources in secondLineDecomposed:
            numberOfResources += 1
            #print(numberOfResources)
            currentActivitySequence.totalResources.append(int(totalResources))
            #print(currentActivitySequence.totalResources)
        currentActivitySequence.numberOfResources = numberOfResources
        next(f)
        thirdLine = f.readline()   # information about starting activities
        thirdLineDecomposed = re.split(" +", thirdLine)
        for IdActivity in thirdLineDecomposed[7:-1]:
            currentActivitySequence.indexStartActivities.append(int(IdActivity)-2)
        #print("indexStartingActivities = " + str(currentActivitySequence.indexStartActivities))
        line = f.readline()
        while line:
            #print(line, end="")
            lineDecomposed = re.split(" +", line)
            if int(lineDecomposed[1]) == 0:
                break
            else:
                currentActivity = activity()
                currentActivity.time = int(lineDecomposed[1])
                currentActivity.requiredResources = [ int(lineDecomposed[2]),int(lineDecomposed[3]),int(lineDecomposed[4]),int(lineDecomposed[5])]
                for IdFollowingActivity in lineDecomposed[7:-1]:
                    if int(IdFollowingActivity) != numberOfActivities+2:    #if the following action is not the last dummy activity
                        currentActivity.indexFollowingActivities.append(int(IdFollowingActivity) - 2)
            currentActivitySequence.activities.append(currentActivity)
            line = f.readline()
        #add indexes to list of activities
        for j in range(len(currentActivitySequence.activities)):
            currentActivitySequence.activities[j].index = j
        #add numberOfPreviousActivities
        for Activity in currentActivitySequence.activities:
            for IndexFollowingActivity in Activity.indexFollowingActivities:
                currentActivitySequence.activities[IndexFollowingActivity].numberOfPreviousActivities += 1
    activitySequences.append(currentActivitySequence)

stateVectorLength = numberOfActivitiesInStateVector + numberOfActivitiesInStateVector * numberOfResources + numberOfResources+ timeHorizon * numberOfResources
#print(stateVectorLength)

# compute decisions: each decision corresponds to a start of an activity in the local reference system (more than one decision can be taken at once)
for i in range(0,numberOfActivitiesInStateVector):
    decisions_indexActivity.append(i)


#--------------------------------------------------------------RANDOM-----------------------------------------------------------------------------
####  GENERATE TRAINING DATA USING RANDOM DECISIONS (WITHOUT USING pool.map) ####
print('######  RANDOM DECISION ON TRAIN ACTIVITY SEQUENCES  ######')
runSimulation_inputs = []
for i in range(numberOfFilesTrain):
    currentRunSimulation_input = runSimulation_input()
    currentRunSimulation_input.activitySequence = activitySequences[indexFilesTrain[i]]
    currentRunSimulation_input.numberOfSimulationRuns = numberOfSimulationRunsToGenerateData
    currentRunSimulation_input.timeDistribution = timeDistribution
    currentRunSimulation_input.purpose = "generateData"
    currentRunSimulation_input.randomDecisionProbability = 1
    currentRunSimulation_input.policyType = None
    # neuralNetworkType needs to be set because states get stored accordingly in states list (vectors vs. matrices)
    currentRunSimulation_input.neuralNetworkType = neuralNetworkType
    currentRunSimulation_input.decisionTool = None
    currentRunSimulation_input.numberOfResources = numberOfResources
    currentRunSimulation_input.numberOfActivitiesInStateVector = numberOfActivitiesInStateVector
    currentRunSimulation_input.stateVectorLength = stateVectorLength
    currentRunSimulation_input.decisions_indexActivity = decisions_indexActivity
    currentRunSimulation_input.rescaleFactorTime = rescaleFactorTime
    currentRunSimulation_input.numberOfActivities = numberOfActivities
    currentRunSimulation_input.timeHorizon = timeHorizon

    runSimulation_inputs.append(currentRunSimulation_input)

pool = mp.Pool(processes=numberOfCpuProcessesToGenerateData)

runSimulation_outputs = pool.map(runSimulation, runSimulation_inputs)
# assign simulation results to activity sequences and append training data

for i in range(numberOfFilesTrain):
    activitySequences[indexFilesTrain[i]].totalDurationMean = runSimulation_outputs[i].totalDurationMean
    activitySequences[indexFilesTrain[i]].totalDurationStandardDeviation = runSimulation_outputs[i].totalDurationStDev
    activitySequences[indexFilesTrain[i]].totalDurationMin = runSimulation_outputs[i].totalDurationMin
    activitySequences[indexFilesTrain[i]].totalDurationMax = runSimulation_outputs[i].totalDurationMax
    activitySequences[indexFilesTrain[i]].luckFactorMean = runSimulation_outputs[i].luckFactorMean
    activitySequences[indexFilesTrain[i]].trivialDecisionPercentageMean = runSimulation_outputs[i].trivialDecisionPercentageMean

    for currentStateActionPair in runSimulation_outputs[i].stateActionPairsOfBestRun:
        states.append(currentStateActionPair.state)
        actions.append(currentStateActionPair.action)


#correspondence best states and actions pairs --> len(states) = len(actions)
#print('state',states)
#print('actions:',actions)


####  TRAIN MODEL USING TRAINING DATA  ####
# look for existing model
print("Train neural network model")
if neuralNetworkType == "1dimensional convnet":
    if importExistingNeuralNetworkModel:
        print("check if a neural network model exists")
        if neuralNetworkModelAlreadyExists:
            print("import neural network model exists")

        else:
            neuralNetworkModel = create1dConvNetNeuralNetworkModel(len(states[0]), len(actions[0]), learningRate)
            #neuralNetworkModel = createNeuralNetworkModel(len(states[0]), len(actionsPossibilities[0]), learningRate)
    else:
        neuralNetworkModel = create1dConvNetNeuralNetworkModel(len(states[0]), len(actions[0]), learningRate)
        #neuralNetworkModel = createNeuralNetworkModel(len(states[0]), len(actionsPossibilities[0]), learningRate)

    neuralNetworkModel.fit({"input": states}, {"targets": actions}, n_epoch=numberOfEpochs, snapshot_epoch=500,
                           show_metric=True, run_id="trainNeuralNetworkModel")

elif neuralNetworkType == "2dimensional convnet":
    # Turn states list into tuples
    states = np.asarray(states)
    #print("states: " + str(states))
    #print("states[0]: " + str(states[0]))
    # Reshape states
    states = states.reshape([-1, len(states[0]), len(states[0]), 1])
    if importExistingNeuralNetworkModel:
        neuralNetworkModelAlreadyExists = False
        print("check if a neural network model exists")
        if neuralNetworkModelAlreadyExists:
            print("import neural network model exists")
        else:
            neuralNetworkModel = create2dConvNetNeuralNetworkModel(len(states[0]), len(actions[0]), learningRate)
    else:
        neuralNetworkModel = create2dConvNetNeuralNetworkModel(len(states[0]), len(actions[0]), learningRate)

    neuralNetworkModel.fit({"input": states}, {"targets": actions}, n_epoch=numberOfEpochs, snapshot_epoch=500,
                           show_metric=True, run_id="trainNeuralNetworkModel")

####  CREATE BENCHMARK WITH RANDOM DECISIONS ALSO WITH TEST ACTIVITY SEQUENCES  ####
print('######  RANDOM DECISION ON TEST ACTIVITY SEQUENCES  ######')
runSimulation_inputs = []
for i in range(numberOfFilesTest):
    currentRunSimulation_input = runSimulation_input()
    currentRunSimulation_input.activitySequence = activitySequences[indexFilesTest[i]]
    currentRunSimulation_input.numberOfSimulationRuns = numberOfSimulationRunsToGenerateData
    currentRunSimulation_input.timeDistribution = timeDistribution
    currentRunSimulation_input.purpose = "testPolicy"
    currentRunSimulation_input.randomDecisionProbability = 1
    currentRunSimulation_input.policyType = None
    currentRunSimulation_input.neuralNetworkType = None
    currentRunSimulation_input.decisionTool = None
    currentRunSimulation_input.numberOfResources = numberOfResources
    currentRunSimulation_input.numberOfActivitiesInStateVector = numberOfActivitiesInStateVector
    currentRunSimulation_input.stateVectorLength = stateVectorLength
    currentRunSimulation_input.decisions_indexActivity = decisions_indexActivity
    currentRunSimulation_input.rescaleFactorTime = rescaleFactorTime
    currentRunSimulation_input.numberOfActivities = numberOfActivities
    currentRunSimulation_input.timeHorizon = timeHorizon

    runSimulation_inputs.append(currentRunSimulation_input)

pool = mp.Pool(processes=numberOfCpuProcessesToGenerateData)

runSimulation_outputs = pool.map(runSimulation, runSimulation_inputs)
# assign simulation results to activity sequences

for i in range(numberOfFilesTest):
    activitySequences[indexFilesTest[i]].totalDurationMean = runSimulation_outputs[i].totalDurationMean
    activitySequences[indexFilesTest[i]].totalDurationStandardDeviation = runSimulation_outputs[i].totalDurationStDev
    activitySequences[indexFilesTest[i]].totalDurationMin = runSimulation_outputs[i].totalDurationMin
    activitySequences[indexFilesTest[i]].totalDurationMax = runSimulation_outputs[i].totalDurationMax
    activitySequences[indexFilesTest[i]].luckFactorMean = runSimulation_outputs[i].luckFactorMean
    activitySequences[indexFilesTest[i]].trivialDecisionPercentageMean = runSimulation_outputs[i].trivialDecisionPercentageMean


#-----------------------------------------------------------------NN------------------------------------------------------------------------------
####  TEST NEURAL NETWORK MODEL ON TRAIN ACTIVITY SEQUENCES  ####
# run simulations with neural network model as decision tool (not possible to use multiprocessing -> apparently is not possible to parallelize processes on GPU)
print('###### NEURAL NETWORK MODEL ON TRAIN ACTIVITY SEQUENCES  ######')
for i in range(numberOfFilesTrain):
    currentRunSimulation_input = runSimulation_input()
    currentRunSimulation_input.activitySequence = activitySequences[indexFilesTrain[i]]
    currentRunSimulation_input.numberOfSimulationRuns = numberOfSimulationRunsToTestPolicy
    currentRunSimulation_input.timeDistribution = timeDistribution
    currentRunSimulation_input.purpose = "testPolicy"
    currentRunSimulation_input.randomDecisionProbability = 0
    currentRunSimulation_input.policyType = "neuralNetworkModel"
    currentRunSimulation_input.neuralNetworkType = neuralNetworkType
    currentRunSimulation_input.decisionTool = neuralNetworkModel
    currentRunSimulation_input.numberOfResources = numberOfResources
    currentRunSimulation_input.numberOfActivitiesInStateVector = numberOfActivitiesInStateVector
    currentRunSimulation_input.stateVectorLength = stateVectorLength
    currentRunSimulation_input.decisions_indexActivity = decisions_indexActivity
    currentRunSimulation_input.rescaleFactorTime = rescaleFactorTime
    currentRunSimulation_input.numberOfActivities = numberOfActivities
    currentRunSimulation_input.timeHorizon = timeHorizon

    currentRunSimulation_output = runSimulation(currentRunSimulation_input)

    activitySequences[indexFilesTrain[i]].totalDurationWithPolicy = currentRunSimulation_output.totalDurationMean



####  TEST NEURAL NETWORK MODEL ON TEST ACTIVITY SEQUENCES  ####
# run simulations with neural network model as decision tool (not possible to use multiprocessing -> apparently is not possible to parallelize processes on GPU)
print('###### NEURAL NETWORK MODEL ON TEST ACTIVITY SEQUENCES  ######')
for i in range(numberOfFilesTest):
    currentRunSimulation_input = runSimulation_input()
    currentRunSimulation_input.activitySequence = activitySequences[indexFilesTest[i]]
    currentRunSimulation_input.numberOfSimulationRuns = numberOfSimulationRunsToTestPolicy
    currentRunSimulation_input.timeDistribution = timeDistribution
    currentRunSimulation_input.purpose = "testPolicy"
    currentRunSimulation_input.randomDecisionProbability = 0
    currentRunSimulation_input.policyType = "neuralNetworkModel"
    currentRunSimulation_input.neuralNetworkType = neuralNetworkType
    currentRunSimulation_input.decisionTool = neuralNetworkModel
    currentRunSimulation_input.numberOfResources = numberOfResources
    currentRunSimulation_input.numberOfActivitiesInStateVector = numberOfActivitiesInStateVector
    currentRunSimulation_input.stateVectorLength = stateVectorLength
    currentRunSimulation_input.decisions_indexActivity = decisions_indexActivity
    currentRunSimulation_input.rescaleFactorTime = rescaleFactorTime
    currentRunSimulation_input.numberOfActivities = numberOfActivities
    currentRunSimulation_input.timeHorizon = timeHorizon

    currentRunSimulation_output = runSimulation(currentRunSimulation_input)

    activitySequences[indexFilesTest[i]].totalDurationWithPolicy = currentRunSimulation_output.totalDurationMean

# Xiaolei: Critical Resource, Shortest Processing Time, shortest sumDuration including successor

# ------------------------------------------------------EVALUATION-----------------------------------------------------------------------------
####  EVALUATION OF RESULTS OF TRAIN ACTIVITY SEQUENCES  ####
sumTotalDurationRandomTrain = 0
#sumTotalDurationWithCriticalResourceTrain = 0
#sumTotalDurationWithShortestProcessingTrain = 0
sumTotalDurationWithNeuralNetworkModelTrain = 0
#sumTotalDurationWithShortestSumDurationTrain = 0

for i in range(numberOfFilesTrain):
    sumTotalDurationRandomTrain += activitySequences[indexFilesTrain[i]].totalDurationMean
    sumTotalDurationRandomTrain = round(sumTotalDurationRandomTrain,4)
    sumTotalDurationWithNeuralNetworkModelTrain += activitySequences[indexFilesTrain[i]].totalDurationWithPolicy
    #sumTotalDurationWithCriticalResourceTrain += activitySequences[indexFilesTrain[i]].totalDurationWithCriticalResource
    #sumTotalDurationWithShortestProcessingTrain += activitySequences[indexFilesTrain[i]].totalDurationWithShortestProcessingTime
    #sumTotalDurationWithShortestSumDurationTrain += activitySequences[indexFilesTrain[i]].totalDurationWithShortestSumDuration

sumTotalDurationRandomTrainRecord.append(sumTotalDurationRandomTrain)
sumTotalDurationWithNeuralNetworkModelTrainRecord.append(sumTotalDurationWithNeuralNetworkModelTrain)
#sumTotalDurationWithCriticalResourceTrainRecord.append(sumTotalDurationWithCriticalResourceTrain)
#sumTotalDurationWithShortestProcessingTrainRecord.append(sumTotalDurationWithShortestProcessingTrain)
#sumTotalDurationWithShortestSumDurationTrainRecord.append(sumTotalDurationWithShortestSumDurationTrain)

####  EVALUATION OF NN RESULTS OF TEST ACTIVITY SEQUENCES  ####
sumTotalDurationRandomTest = 0
sumTotalDurationWithNeuralNetworkModelTest = 0
sumTotalDurationWithCriticalResourceTest = 0
sumTotalDurationWithShortestProcessingTest = 0
sumTotalDurationWithShortestSumDurationTest = 0

for i in range(numberOfFilesTest):
    sumTotalDurationRandomTest += activitySequences[indexFilesTest[i]].totalDurationMean
    sumTotalDurationRandomTest = round(sumTotalDurationRandomTest,4)
    sumTotalDurationWithNeuralNetworkModelTest += activitySequences[indexFilesTest[i]].totalDurationWithPolicy
    #sumTotalDurationWithCriticalResourceTest += activitySequences[indexFilesTest[i]].totalDurationWithCriticalResource
    #sumTotalDurationWithShortestProcessingTest += activitySequences[indexFilesTest[i]].totalDurationWithShortestProcessingTime
    #sumTotalDurationWithShortestSumDurationTest += activitySequences[indexFilesTest[i]].totalDurationWithShortestSumDuration


sumTotalDurationRandomTestRecord.append(sumTotalDurationRandomTest)
sumTotalDurationWithNeuralNetworkModelTestRecord.append(sumTotalDurationWithNeuralNetworkModelTest)
#sumTotalDurationWithCriticalResourceTestRecord.append(sumTotalDurationWithCriticalResourceTest)
#sumTotalDurationWithShortestProcessingTestRecord.append(sumTotalDurationWithShortestProcessingTest)
#sumTotalDurationWithShortestSumDurationTestRecord.append(sumTotalDurationWithShortestSumDurationTest)



print("sumTotalDurationRandomTrain = " + str(sumTotalDurationRandomTrain))
print("sumTotalDurationWithNeuralNetworkModelTrain = " + str(sumTotalDurationWithNeuralNetworkModelTrain))
#print("sumTotalDurationWithCriticalResourceTrain = " + str(sumTotalDurationWithCriticalResourceTrain))
#print("sumTotalDurationWithShortestProcessingTrain = " + str(sumTotalDurationWithShortestProcessingTrain))
#print("sumTotalDurationWithShortestSumDurationTrain = " + str(sumTotalDurationWithShortestSumDurationTrain))
print("sumTotalDurationRandomTest = " + str(sumTotalDurationRandomTest))
print("sumTotalDurationWithNeuralNetworkModelTest = " + str(sumTotalDurationWithNeuralNetworkModelTest))
#print("sumTotalDurationWithCriticalResourceTest = " + str(sumTotalDurationWithCriticalResourceTest))
#print("sumTotalDurationWithShortestProcessingTest = " + str(sumTotalDurationWithShortestProcessingTest))
#print("sumTotalDurationWithShortestSumDurationTest = " + str(sumTotalDurationWithShortestSumDurationTest))


# compute computation time
t_end = time.time()
t_computation = t_end - t_start
print("t_computation = " + str(t_computation))