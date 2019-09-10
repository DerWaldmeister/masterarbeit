import numpy as np
import random

currentState_readyToStartActivities = []
currentState_readyToStartActivitiesV = []
stateVectorLength = 34

# Create a vector
#currentState_readyToStartActivities = np.zeros([stateVectorLength])
currentState_readyToStartActivitiesV = np.zeros((stateVectorLength, 1))
for i in range(len(currentState_readyToStartActivitiesV)):
    currentState_readyToStartActivitiesV[i, 0] = i + 1

#print("currentState_readyToStartActivities: " + str(currentState_readyToStartActivities))
print("currentState_readyToStartActivitiesV: " + str(currentState_readyToStartActivitiesV))

# Transpose the vector
currentState_readyToStartActivitiesVT = currentState_readyToStartActivitiesV.reshape(-1, stateVectorLength)
print("currentState_readyToStartActivitiesVT: " + str(currentState_readyToStartActivitiesVT))

# Multiply the vector with its transposed vector
currentState_readyToStartActivitiesM = np.matmul(currentState_readyToStartActivitiesV, currentState_readyToStartActivitiesVT)
print("currentState_readyToStartActivitiesM: " + str(currentState_readyToStartActivitiesM))





