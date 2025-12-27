from ObjectClassifier import ObjectDataset, ClassifierModel
import Configuration
import os
import numpy as np

# Set the environment variable to suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def trainAndSave():
    modelConfig = Configuration.config["model"]
    datasetConfig = Configuration.config["dataset"]

    batchSize = datasetConfig["BATCH_SIZE"]
    roadSignDataset = ObjectDataset.ObjectDataset(
        datasetDir=datasetConfig["DATASET_DIR"],
        imageWidth=datasetConfig["IMAGE_WIDTH"],
        imageHeight=datasetConfig["IMAGE_HEIGHT"],
        batchSize=datasetConfig["BATCH_SIZE"],
    )

    # Get class names from dataset
    classNames = roadSignDataset.getClassNames()
    nClasses = len(classNames)

    np.savetxt(datasetConfig["CLASS_NAME_FILE"], classNames, fmt="%s")

    trainData, validData, testData = roadSignDataset.getData(
        trainRatio=0.8, validRatio=0.2
    )

    print(f"\nTrain data size={len(list(trainData)) * batchSize}")
    print(f"test data size={len(list(testData)) * batchSize}")
    print(f"and validation data size={len(list(validData)) * batchSize}")
    print(f"Number of classes = {nClasses}\n")

    roadSignDataset.plotClassDistribution()
    roadSignDataset.plotExamplesFromDataset(7)

    model = ClassifierModel.ClassifierModel(
        inputWidth=datasetConfig["IMAGE_HEIGHT"],
        inputHeight=datasetConfig["IMAGE_HEIGHT"],
        nClasses=nClasses,
    )
    model.plotModel()
    model.train(
        nEpochs=modelConfig["EPOCHS"],
        trainDataset=trainData,
        validDataset=validData,
        testDataset=testData,
    )
    model.plotTrainingHistory()
    model.saveModel(filePath=modelConfig["MODEL_PATH"])


if __name__ == "__main__":
    trainAndSave()
