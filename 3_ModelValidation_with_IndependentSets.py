"""
CNN model build with Keras 
Created date : 18/12/18 
author : MOLA lIN
"""
###  ----------------------------------------------------------------------
import os
import sys
os.environ ["CUDA_VISIBLE_DEVICES"] = "0"  # assign specific graphic card
os.environ ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # ignore system informaion and error

import numpy as np
import matplotlib
matplotlib.use( "Agg" )
import matplotlib.pyplot as plt
from datetime import datetime as DT

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import BatchNormalization as B_nor
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K

###  ----------------------------------------------------------------------
###  Build save_train_history to save training processes
###  ----------------------------------------------------------------------
def save_train_history ( train_history, train, validation, name ):
    plt.plot ( train_history.history[train] )
    plt.plot ( train_history.history[validation] )
    plt.title ( 'Train History' )
    plt.ylabel ( train )
    plt.xlabel ( 'Epoch' )
    plt.legend ( ['train', 'validation'], loc = 'upper left' )
    name = "./" + name + ".png"
    plt.savefig( name, bbox_inches = "tight" )
    plt.close()

def MCC(y_true, y_pred): # Matthews correlation coefficient
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

###  ----------------------------------------------------------------------
###  Load training data and its label
###  and do the input-format convertion
###  ----------------------------------------------------------------------
# load testing data
testing_set = sys.argv[1] 
test_sample = np.load ( testing_set )
test_sample = test_sample.reshape ( (test_sample.shape[0], 100, 100, 1) ) # no. of samples, x pixels, y pixels, no. of files

# load labels of testing data
testing_label = sys.argv[2] 
test_label = np.load ( testing_label )
test_label_compare = test_label
test_label = np_utils.to_categorical ( test_label )

# load testing data's title
filename_list = sys.argv[3] 
test_sample_title = np.load ( filename_list )

input_save_model = sys.argv[4] 
time = DT.now() #system time for output file name
time = str(time.year) + str(time.month) + str(time.day) + str(time.hour) + str(time.minute)
out=open( os.path.splitext(os.path.basename(testing_set))[0] + "_predicted_result_"+time,"w" )
print("Training model from "+input_save_model, file=out)
print("Indenpendent set from "+testing_set, file=out)

###  ----------------------------------------------------------------------
###  build CNN model
###  ----------------------------------------------------------------------
model = Sequential()

### Conv 1
model.add ( Conv2D ( filters = 64, kernel_size = ( 5, 5 ), input_shape = ( 100, 100, 1 ) ) )
model.add ( B_nor ( axis = 2, epsilon = 1e-5 ) )
model.add ( MaxPooling2D ( pool_size = (2, 2) , padding = "same" ) )

### Conv 2
model.add ( Conv2D ( filters = 64, kernel_size = ( 3, 3 ) ) )
model.add ( B_nor ( axis = 2, epsilon = 1e-5 ) )
model.add ( MaxPooling2D ( pool_size = (2, 2) , padding = "same" ) )

### Conv 3
model.add ( Conv2D ( filters = 64, kernel_size = ( 3, 3 ) ) )
model.add ( B_nor ( axis = 2, epsilon = 1e-5 ) )
model.add ( MaxPooling2D ( pool_size = (2, 2) , padding = "same" ) )

### Fully connect
model.add ( Flatten() )
### hiden layers
model.add ( Dense ( 1000, activation = "relu" ) )
model.add ( Dense ( 600, activation = "relu" ) )
model.add ( Dense ( 80, activation = "relu" ) )
model.add ( Dense ( 12, activation = "softmax" ) )

###  ----------------------------------------------------------------------
###  load weight from previous training / Best performance
###  ----------------------------------------------------------------------
try:
  model.load_weights ( input_save_model )
  print ( "Successfully loading previous BEST training weights." )
except:
  print ( "Failed to load previous data, training new model below." )

###  ----------------------------------------------------------------------
###  compile and run the model with input
###  ----------------------------------------------------------------------
Adam = Adam ( lr = 1e-4 )
model.compile ( loss = 'categorical_crossentropy',
                optimizer = Adam, metrics = ['accuracy',MCC] )

###  ----------------------------------------------------------------------
###  Independent Testing Results
###  ----------------------------------------------------------------------
scores = model.evaluate ( test_sample, test_label, verbose = 1 )
print ( "Independent test:\tAccuracy\t%.3f\tMCC\t%.3f\n" % ( scores[1] , scores[2]) , file=out)

###  ----------------------------------------------------------------------
###  Use trained model to do the prediction
###  ----------------------------------------------------------------------
Prediction = model.predict_classes ( test_sample )
num = 0
start = 1
print ( "Predict-result :" , file=out)
for result in Prediction :
    label_from_title = str ( test_sample_title[num] ).split("-")
    if int(label_from_title[2]) != result :
        print ("%i\tSample Name:\t%s\tActual:\t%i\tPredict:\t%i" 
               %( start, test_sample_title[num], int(label_from_title[2]),  result ) , file=out)
        start += 1
    num += 1

###  ----------------------------------------------------------------------
###  confusion matrix
###  ----------------------------------------------------------------------
import pandas as pd
confusion_matrix = pd.crosstab(test_label_compare, Prediction,rownames=['Actual'], colnames=['Predicted'], margins=True)  # return dataframe (df) formate
print ( "\n\nConfusion Matrix:\n%s" % confusion_matrix)
print ( "\n\nConfusion Matrix:\n%s" % confusion_matrix, file=out)

out.close()
