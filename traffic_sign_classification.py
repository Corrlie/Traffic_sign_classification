import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import pickle

with open("train.p", mode='rb') as training_data:
    train = pickle.load(training_data)
with open("valid.p", mode='rb') as validation_data:
    valid = pickle.load(validation_data)
with open("test.p", mode='rb') as testing_data:
    test = pickle.load(testing_data)

X_train, y_train = train['features'], train['labels']
X_validation, y_validation = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)

# Konwersja na skalę szarości
X_train_gray = np.sum(X_train/3, axis = 3, keepdims = True)
X_validation_gray = np.sum(X_validation/3, axis = 3, keepdims = True)
X_test_gray = np.sum(X_test/3, axis = 3, keepdims = True)

# NNormalizacja
X_train_gray_norm = (X_train_gray - 128)/128
X_validation_gray_norm = (X_validation_gray - 128)/128
X_test_gray_norm = (X_test_gray - 128)/128

from tensorflow.keras import datasets, layers, models, regularizers # import potrzebnych pakietów

CNN = models.Sequential() # utworzenie modelu sekwencyjnego

# tworzenie kolejnych warstw zbudowanego modelu
CNN.add(layers.Conv2D(4, (7,7), activation = "relu", input_shape=(32,32,1)))
CNN.add(layers.AveragePooling2D())

CNN.add(layers.Dropout(0.4))

CNN.add(layers.Conv2D(13, (7,7), activation = "relu"))
CNN.add(layers.AveragePooling2D())

CNN.add(layers.Flatten())

CNN.add(layers.Dense(120, activation="relu", activity_regularizer=regularizers.l2(0.01)))
CNN.add(layers.Dense(84, activation= "relu", activity_regularizer=regularizers.l2(0.01)))
CNN.add(layers.Dense(43, activation = "softmax"))
CNN.summary() # podsumowanie modelu
# print(CNN.summary())

CNN.compile(optimizer = 'Adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

history = CNN.fit(X_train_gray_norm,
                 y_train,
                 batch_size = 100,
                 epochs = 10,
                 verbose = 1,
                 validation_data = (X_validation_gray_norm, y_validation))

score = CNN.evaluate(X_test_gray_norm, y_test)

print('Test Accuracy: {}'.format(score[1]))

# zapis do wartości trafności pliku
plik = open('score.txt', 'a')
plik.write('\nTest Accuracy: {}'.format(score[1]))
plik.close()

history.history.keys()
print(history.history.keys())

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# tworzenie wykresów accuracy/loss
epochs = np.arange(1,(len(accuracy)+1))

plt.plot(epochs, loss, 'r', label = 'Training Loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')
plt.legend()
plt.xlabel('Epchos')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.show()

plt.plot(epochs, accuracy, 'r', label = 'Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label = 'Validation accuracy')
plt.legend()
plt.xlabel('Epchos')
plt.ylabel('Accuracy')
plt.title('Training and Validation accuracy')
plt.show()

predicted_classes = CNN.predict_classes(X_test_gray_norm)
y_true = y_test

# tworzenie macierzy błędów
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, predicted_classes)

plt.figure(figsize = (25, 25))
sns.heatmap(cm, annot = True)

# tworzenie siatki obrazów
L = 5
W = 5

fig, axes = plt.subplots(L, W, figsize = (12, 12))
axes = axes.ravel()

for i in np.arange(0, L*W):
    axes[i].imshow(X_test[i])
    axes[i].set_title('Prediction = {}\n True = {}'.format(predicted_classes[i], y_true[i]))
    axes[i].axis('off')

plt.subplots_adjust(wspace = 1)
plt.show()

