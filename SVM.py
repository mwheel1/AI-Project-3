
#File:    SVM.py
#Author:  Matthew Wheeler
#Date:    05/04/2016
#Section: 02
#E-mail:  mwheel1@umbc.edu
#Description: A python machine learning program that classifies
# a user inputted image file into one of 5 categories.

from sklearn import svm
from PIL import Image
import numpy as np
import sys

INPUT_ARGS = 2

def main():
    if (len(sys.argv) != INPUT_ARGS):
        print("Invalid number of Arguments. Check Syntax.")
        print("Syntax: SVM.py <input img>")
        print("Where <input img> = File path to single image")
        print("Exiting...")
        exit()
    else:
        print("Executing Program...")

        print("Processing Input Image...")
        inputFile = str(sys.argv[1])
        try:
            img = Image.open(inputFile)
            img = img.convert('L')
            img = np.array(img)
            img = img.ravel().reshape(1,-1)
            print("Input Image Processed.")
        except IOError:
            print("Input File Error.")
            print("Verify file is .jpg and Check Path: "+inputFile)
            print("Exiting...")
            exit()

        print("Loading SVM...")
        try:
            X = np.load("data.npy")
            Y = np.load("labels.npy")
            clf = svm.SVC()
            clf.fit(X,Y)
            print("SVM Loaded.")
        except IOError:
            print("Data File Not Found.")
            print("Make sure there is a data.npy and label.npy in the program directory.")
            print("Run getData.py to create the files.")
            print("Exiting...")
            exit()
            
        print("Classifying Image...")
        result = clf.predict(img)[0]
        if (result == 1):
            print("Classification: SMILEY FACE")
        elif (result == 2):
            print("Classification: TOP HAT")
        elif (result == 3):
            print("Classification: OCTOTHORPE")
        elif (result == 4):
            print("Classification: HEART")
        elif (result == 5):
            print("Classification: DOLLAR SIGN")
    return

main()
