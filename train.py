import pandas as pd 
import collections
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
from sklearn.utils import class_weight
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

# Function that prints metrics for given predictions and true values 
def printMetrics(y_true,y_pred):
    print("Accuracy: ",accuracy_score(y_true,y_pred))
    print("Recall: ",recall_score(y_true,y_pred))
    print("Precision score: ", precision_score(y_true,y_pred))
    print("F1 Score: ",f1_score(y_true,y_pred))

# Load the data
data_train = pd.read_csv('train_clean.csv')
data_test = pd.read_csv('test_clean.csv')

# Transform into tf-idf vectors
tfidf = TfidfVectorizer( analyzer='word', max_features=40000)
X_train = tfidf.fit_transform(data_train['Text']).astype('float16')
X_test = tfidf.transform(data_test['Text']).astype('float16')

# Transform the targets from dataframe to list
Y_train = data_train['Target'].tolist()
Y_test = data_test['Target'].tolist()

def build_model():
    model = Sequential()
    model.add(Dense(256, input_dim=40000, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(3000, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2600, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2200, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(900, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

# Add class weigths to the model since we are dealing with an unbalanced dataset 
class_weights = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)
print("Class weights:")
print(class_weights)

# Fit and predict
estimator = KerasClassifier(build_fn=build_model, epochs=30,batch_size=32)
estimator.fit(X_train,Y_train, class_weight=class_weights)
Y_pred = estimator.predict(X_test)

printMetrics(Y_test, Y_pred)

