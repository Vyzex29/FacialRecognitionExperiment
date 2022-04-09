import PackageInstaller
import PicturePipeline
import TFRecognition.HaarCascade.HaarCascadesFaceTrain as HaarTrain
import TFRecognition.HaarCascade.HaarRecognition as HaarRec
import TensorRecognition as TensorRec
import Configuration.Config as config
import TensorTrain


def printMenu():
    print("Menu:")
    print("1: Run Picture Pipeline")
    print("2: Initiate a learning algorithm")
    print("3: Run facial recognition + attendance")
    print("4: Exit")

def printAlgorithmMenu():
    print("Choose learning algorithm")
    print("1. Haar Cascade")
    print("2. Tensorflow CNN")
    print("3. Caffe")
    print("4. Exit to previous menu")

def printRecognitionMenu():
    print("Choose Recognition algorithm")
    print("1. Haar Cascade")
    print("2. Tensorflow CNN")
    print("3. Caffe")
    print("4. Exit to previous menu")

def userMenuInput():
    printMenu()
    showMenu = True
    menuItem = input()
    if menuItem == "1":
        print("Controls: \n Press Q, while having the pictureBox opened to process next image")
        PicturePipeline.runPicturePipeline()
    elif menuItem == "2":
        showAlgorithmMenu = True
        while showAlgorithmMenu:
            printAlgorithmMenu()
            algorithmMenu = input()
            if algorithmMenu == "1":
                print("Started learning Haar")
                HaarTrain.Train()
                print("File Created:" + config.HAAR_TRAINED_IMAGE_FILENAME)
                showAlgorithmMenu = False
            elif algorithmMenu == "2":
                print("Started learning Tensorflow")
                TensorTrain.Train()
                showAlgorithmMenu = False
            elif algorithmMenu == "3":
                print("Started learning Caffe")
                showAlgorithmMenu = False
            elif algorithmMenu == "4":
                showAlgorithmMenu = False
            else:
                print("Choose from the algorithm menu")
    elif menuItem == "3":
        showRecognitionMenu = True
        while showRecognitionMenu:
            printRecognitionMenu()
            recognitionMenu = input()
            if recognitionMenu == "1":
                print("Haar Recognition Started")
                HaarRec.Start()
                showRecognitionMenu = False
            elif recognitionMenu == "2":
                print("Tensor Recognition Started")
                TensorRec.Start()
                showRecognitionMenu = False
            elif recognitionMenu == "3":
                print("Caffe Recognition started")
                showRecognitionMenu = False
            elif showRecognitionMenu == "4":
                showRecognitionMenu = False
            else:
                print("Choose from the recognition menu")
    elif menuItem == "4":
        showMenu = False
    elif menuItem == "5":
        PicturePipeline.CreateYOLOLabelClasses()
    else:
       print("Please choose a menu item")
    return showMenu


isRunning = True
while isRunning:
    isRunning = userMenuInput()

