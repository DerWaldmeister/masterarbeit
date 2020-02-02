import time
import os
import glob
import numpy as np
import random
import re
import tensorflow as tf
from env import runSimulation, runSimulation_input, activitySequence, activity
from convnet_1d import create1dConvNetNeuralNetworkModel
from convnet_2d import create2dConvNetNeuralNetworkModel
from combined_convnet_2d import createCombined2dConvNetNeuralNetworkModelForFutureResourceUtilisation
from combined_convnet_1d import createCombined1dConvNetNeuralNetworkModelForFutureResourceUtilisation
import multiprocessing as mp
from openpyxl import Workbook
from openpyxl.styles import Border, Alignment, Side
from randomize_train_validate_test_indices import randomizeTrainValidateTestIndeces
from datetime import datetime


t_start = time.time()

# user defined parameters
# problem parameters
timeDistribution = "deterministic"    # deterministic, exponential, uniform_1, uniform_2, ...

# file name flag
fileNameLabel = 'main_RG30'

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
neuralNetworkType = "2dimensional convnet" # 1dimensional convnet, 2dimensional convnet, 1dimensional combined convnet, 2dimensional combined convnet
# for 1dimensional convnet and 2dimesnional convnet futureResourceUtilisation wont be used
useFutureResourceUtilisation = False
if neuralNetworkType == "1dimensional combined convnet" or neuralNetworkType == "2dimensional combined convnet":
    useFutureResourceUtilisation = True

# train parameter
generateNewTrainTestValidateSets = False
importExistingNeuralNetworkModel = False
neuralNetworkModelAlreadyExists = False
numberOfEpochs = 3000 #walk entire samples
epochsTrainingInterval = 100
# learning rate
learningRate = 0.00005

# paths
relativePath = os.path.dirname(__file__)
absolutePathProjects = relativePath + "/database/RG30_Newdata/"

# other parameters
np.set_printoptions(precision=4)    # print precision of numpy variables

# Activity Sequence variables
# initialise variables
numberOfActivities = None
numberOfResources = None
activitySequences = []
decisions_indexActivity = []
decisions_indexActivityPowerset = []

# read all activity sequences from database
absolutePathProjectsGlob = absolutePathProjects + "*.txt"
files = sorted(glob.glob(absolutePathProjectsGlob))

# divide all activity sequences in training and test set
numberOfFiles = len(files)

# call randomizeTrainValidateTestIndeces to generate a new train validate test split or use an already created split
indexFilesTrain, indexFilesValidate, indexFilesTest = randomizeTrainValidateTestIndeces(numberOfFiles, generateNewTrainTestValidateSets , fileNameLabel)

numberOfFilesTrain = len(indexFilesTrain)
numberOfFilesValidate = len(indexFilesValidate)
numberOfFilesTest = len(indexFilesTest)

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


# states, action and futureResourceUtilisationMatrices of training set
states = []
actions = []
futureResourceUtilisationMatrices = []
# states, action and futureResourceUtilisationMatrices of validation set
statesValidationSet = []
actionsValidationSet = []
futureResourceUtilisationMatricesValidationSet = []
sumTotalDurationRandomValidateRecord = []
#sumTotalDurationWithNeuralNetworkModelValidateRecord = []
sumTotalDurationsPerEpochsWithNeuralNetworkModelValidateRecords = []
#sumTotalDurationWithCriticalResourceValidateRecord = []
#sumTotalDurationWithShortestProcessingValidateRecord = []
#sumTotalDurationWithShortestSumDurationValidateRecord = []
sumTotalDurationRandomTrainRecord = []
#sumTotalDurationWithNeuralNetworkModelTrainRecord = []
#sumTotalDurationWithCriticalResourceTrainRecord = []
#sumTotalDurationWithShortestProcessingTrainRecord = []
#sumTotalDurationWithShortestSumDurationTrainRecord = []
#sumTotalDurationRandomTestRecord = []
#sumTotalDurationWithNeuralNetworkModelTestRecord = []
#sumTotalDurationWithCriticalResourceTestRecord = []
#sumTotalDurationWithShortestProcessingTestRecord = []
#sumTotalDurationWithShortestSumDurationTestRecord = []

#NEW
minDurationPerActivitySequenceTrainRecord = []
minDurationPerActivitySequenceValidateRecord = []
meanDurationPerActivitySequenceTrainRecord = []
meanDurationPerActivitySequenceValidateRecord = []
percentageImprovementMinToMeanValidateRecord = []

'''
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
    currentRunSimulation_input.useFutureResourceUtilisation = useFutureResourceUtilisation

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

    minDurationPerActivitySequenceTrainRecord.append([indexFilesTrain[i], activitySequences[indexFilesTrain[i]].totalDurationMin])
    
    

    # saving validation set states, actions and futureResourceUtilisationMatrices
    #for currentStateActionPair in runSimulation_outputs[i].stateActionPairsOfBestRun:
    #    states.append(currentStateActionPair.state)
    #    actions.append(currentStateActionPair.action)
        # NEW:
    #    futureResourceUtilisationMatrices.append(currentStateActionPair.futureResourceUtilisationMatrix)

'''
#correspondence best states and actions pairs --> len(states) = len(actions)
#print('state',states)
#print('actions:',actions)

####  CREATE BENCHMARK WITH RANDOM DECISIONS ALSO WITH VALIDATION ACTIVITY SEQUENCES  ####
print('######  RANDOM DECISION ON VALIDATE ACTIVITY SEQUENCES  ######')
runSimulation_inputs = []
for i in range(numberOfFilesValidate):
    currentRunSimulation_input = runSimulation_input()
    currentRunSimulation_input.activitySequence = activitySequences[indexFilesValidate[i]]
    currentRunSimulation_input.numberOfSimulationRuns = numberOfSimulationRunsToGenerateData
    currentRunSimulation_input.timeDistribution = timeDistribution
    currentRunSimulation_input.purpose = "generateData"
    currentRunSimulation_input.randomDecisionProbability = 1
    currentRunSimulation_input.policyType = None
    currentRunSimulation_input.neuralNetworkType = neuralNetworkType
    currentRunSimulation_input.decisionTool = None
    currentRunSimulation_input.numberOfResources = numberOfResources
    currentRunSimulation_input.numberOfActivitiesInStateVector = numberOfActivitiesInStateVector
    currentRunSimulation_input.stateVectorLength = stateVectorLength
    currentRunSimulation_input.decisions_indexActivity = decisions_indexActivity
    currentRunSimulation_input.rescaleFactorTime = rescaleFactorTime
    currentRunSimulation_input.numberOfActivities = numberOfActivities
    currentRunSimulation_input.timeHorizon = timeHorizon
    currentRunSimulation_input.useFutureResourceUtilisation = useFutureResourceUtilisation

    runSimulation_inputs.append(currentRunSimulation_input)

pool = mp.Pool(processes=numberOfCpuProcessesToGenerateData)

runSimulation_outputs = pool.map(runSimulation, runSimulation_inputs)
# assign simulation results to activity sequences

for i in range(numberOfFilesValidate):
    activitySequences[indexFilesValidate[i]].totalDurationMean = runSimulation_outputs[i].totalDurationMean
    activitySequences[indexFilesValidate[i]].totalDurationStandardDeviation = runSimulation_outputs[i].totalDurationStDev
    activitySequences[indexFilesValidate[i]].totalDurationMin = runSimulation_outputs[i].totalDurationMin
    activitySequences[indexFilesValidate[i]].totalDurationMax = runSimulation_outputs[i].totalDurationMax
    activitySequences[indexFilesValidate[i]].luckFactorMean = runSimulation_outputs[i].luckFactorMean
    activitySequences[indexFilesValidate[i]].trivialDecisionPercentageMean = runSimulation_outputs[i].trivialDecisionPercentageMean

    # List min and mean values for each activitySequence
    minDurationPerActivitySequenceValidateRecord.append(
        [indexFilesValidate[i], activitySequences[indexFilesValidate[i]].totalDurationMin])
    meanDurationPerActivitySequenceValidateRecord.append([indexFilesValidate[i], activitySequences[indexFilesValidate[i]].totalDurationMean])
    # Calculate percentage improvement for each activitySequence
    percentageImprovementMinToMeanValidateRecord.append([indexFilesValidate[i], round(100*(1 - (activitySequences[indexFilesValidate[i]].totalDurationMin/activitySequences[indexFilesValidate[i]].totalDurationMean)),4)])

sumPercentageImprovement = 0
#Calculate mean percentage improvement over all activitySequences
for i in range(numberOfFilesValidate):
    sumPercentageImprovement = sumPercentageImprovement + percentageImprovementMinToMeanValidateRecord[i][1]

meanPercentageImprovement = round(sumPercentageImprovement/numberOfFilesValidate, 3)


print("minDurationPerActivitySequenceValidateRecord: " + str(minDurationPerActivitySequenceValidateRecord))
print("meanDurationPerActivitySequenceValidateRecord " + str(meanDurationPerActivitySequenceValidateRecord))
print("percentageImprovementMinToMeanValidateRecord " + str(percentageImprovementMinToMeanValidateRecord))
print("meanPercentageImprovement " + str(meanPercentageImprovement))