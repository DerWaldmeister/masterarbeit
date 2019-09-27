import random
import pickle


def randomizeTrainValidateTestIndeces(numberOfFiles, generateNewTrainTestValidateSets):

    indexFilesTrainPickleFile = 'indexFilesTrainPpsLib.pk'
    indexFilesValidatePickleFile = 'indexFilesValidatePpsLib.pk'
    indexFilesTestPickleFile = 'indexFilesTestPpsLib.pk'

    #generate new train test validation sets
    if generateNewTrainTestValidateSets:
        # psplib J30 has 480 files
        #numberOfFiles = 480
        indexFiles = list(range(0, numberOfFiles))

        percentageOfFilesTest = 0.2
        percentageOfFilesValidate = 0.2

        #indexFilesTrain = []
        indexFilesTest = []
        indexFilesValidate = []

        numberOfFilesTest = round(numberOfFiles * percentageOfFilesTest)  # 96
        numberOfFilesValidate = round(numberOfFiles * percentageOfFilesValidate) # 96
        #numberOfFilesTrain = numberOfFiles - numberOfFilesTest - numberOfFilesValidate # 288


        for i in range(numberOfFilesTest):
            randomIndexTest = random.randrange(0, len(indexFiles))

            indexFilesTest.append(indexFiles[randomIndexTest])
            del indexFiles[randomIndexTest]

        for i in range(numberOfFilesValidate):
            randomIndexValidate = random.randrange(0, len(indexFiles))
            #while randomIndexValidate == randomIndexTest:
            #    randomIndexValidate = random.randrange(0, len(indexFiles))
            indexFilesValidate.append(indexFiles[randomIndexValidate])
            del indexFiles[randomIndexValidate]

        indexFilesTrain = indexFiles

        # store indexLists in pickle files:
        # open a pickle file
        with open(indexFilesTrainPickleFile, 'wb') as fi:
            # dump data in pickle file
            pickle.dump(indexFilesTrain, fi)

        with open(indexFilesValidatePickleFile, 'wb') as fi:
            # dump data in pickle file
            pickle.dump(indexFilesValidate, fi)

        with open(indexFilesTestPickleFile, 'wb') as fi:
            # dump data in pickle file
            pickle.dump(indexFilesTest, fi)

        # Open pickle files so they can be returned
        with open(indexFilesTrainPickleFile, 'rb') as fi:
            indexFilesTrain = pickle.load(fi)

        with open(indexFilesValidatePickleFile, 'rb') as fi:
            indexFilesValidate = pickle.load(fi)

        with open(indexFilesTestPickleFile, 'rb') as fi:
            indexFilesTest = pickle.load(fi)


    # just return the already existing train validate test sets from the pickle files
    else:

        with open(indexFilesTrainPickleFile, 'rb') as fi:
            indexFilesTrain = pickle.load(fi)

        with open(indexFilesValidatePickleFile, 'rb') as fi:
            indexFilesValidate = pickle.load(fi)

        with open(indexFilesTestPickleFile, 'rb') as fi:
            indexFilesTest = pickle.load(fi)



    indexFilesTest.sort()
    indexFilesValidate.sort()
    indexFilesTrain.sort()

    print("indexFilesTest: " + str(indexFilesTest))
    print("len(indexFilesTest): " + str(len(indexFilesTest)))
    print("indexFilesValidate: " + str(indexFilesValidate))
    print("len(indexFilesValidate): " + str(len(indexFilesValidate)))
    print("indexFilesTrain: " + str(indexFilesTrain))
    print("len(indexFilesTrain) " + str(len(indexFilesTrain)))

    return indexFilesTrain, indexFilesValidate, indexFilesTest
