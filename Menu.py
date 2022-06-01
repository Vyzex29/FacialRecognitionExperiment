import PackageInstaller
import PicturePipeline
import TFRecognition.HaarCascade.HaarCascadesFaceTrain as HaarTrain
import TFRecognition.HaarCascade.HaarRecognition as HaarRec
import TensorRecognition as TensorRec
import TensorRecognitionImages as TensorRecImg
import Configuration.Config as config
import TensorTrain
import YoloImageRecognition as yir

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
                print("Started learning Yolov5")
                showAlgorithmMenu = False
            elif algorithmMenu == "4":
                print("Started learning Siamese")
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
                print("Tensor images Recognition started")
                TensorRecImg.Start()
                showRecognitionMenu = False
            elif showRecognitionMenu == "4":
                showRecognitionMenu = False

            else:
                print("Choose from the recognition menu")
    elif menuItem == "4":
        showMenu = False
    elif menuItem == "5":
        yir.Start()
    elif menuItem == "6":
        TensorRecImg.StartTest()
    else:
       print("Please choose a menu item")
    return showMenu


isRunning = True
while isRunning:
    isRunning = userMenuInput()

