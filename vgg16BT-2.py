from keras.applications.vgg16 import VGG16
#vggmodel = VGG16(weights='imagenet', include_top=True,)
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.layers import Dense,Flatten
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report
from keras import optimizers
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, models
from keras.layers import Dense
import matplotlib.pyplot as plt
from glob import glob
import cv2
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def carica_imgs(dataset):
  images = []
  for img_path in dataset:
       img0 = cv2.imread(img_path)
       img0 = cv2.resize(img0, (128,128), interpolation = cv2.INTER_AREA)
       img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
       images.append(img0)
  return images

def download_Dataset():
    all_images_train = glob("C:/Users/ilcai/Desktop/ProgettoML/Dataset/Training/*/*.jpg")
    labels_train = converti(all_images_train)
    all_images_test = glob("C:/Users/ilcai/Desktop/ProgettoML/Dataset/Testing/*/*.jpg")
    labels_test = converti(all_images_test)
    return all_images_train,labels_train,all_images_test,labels_test

def converti(all_images):
  labels = []
  for img in all_images:
    if img.split("\\")[1] == 'glioma':
      labels.append(1)
    elif img.split("\\")[1] == 'meningioma':
      labels.append(2)
    elif img.split("\\")[1] == 'pituitary':
      labels.append(3)
    elif img.split("\\")[1] == 'no_tumor' or img.split("\\")[1] == 'notumor':
      labels.append(0)
  return labels

train_path,train_labels,test_path,test_labels = download_Dataset()
#train_path, test_path,train_labels,test_labels = train_test_split(images,labels, test_size=0.33, random_state=42)
len(test_images)

train_images = carica_imgs(train_path)
test_images = carica_imgs(test_path)
train_images_np = np.array(train_images)
train_labels_np = np.array(train_labels)
test_images_np = np.array(test_images)
test_labels_np = np.array(test_labels)
train_labels_np = train_labels_np.reshape(len(train_labels_np),1)
test_labels_np = test_labels_np.reshape(len(test_labels_np),1)


vggmodel = VGG16(weights='imagenet', include_top=False,input_shape= (128,128,3))
vggmodel.trainable = False

vggmodel.summary()
for layers in (vggmodel.layers)[:19]:
    layers.trainable = False

#flatten_layer = Flatten()
#dense_layer_1 = Dense(50, activation='relu')
#dense_layer_2 = Dense(50, activation='relu')
#prediction_layer = Dense(4,activation = 'softmax')


from keras import *
model_final = models.Sequential()
model_final.add(Input(shape = (128,128,3)))
model_final.add(vggmodel)
model_final.add(Flatten())
model_final.add(Dense(50,activation='relu'))
model_final.add(Dense(50,activation='relu'))
model_final.add(Dense(4,activation = 'softmax'))

#X= vggmodel.layers[-2].output
#vggmodel.input
#predictions = Dense(2, activation="softmax")(X)
#model_final = Model(input = vggmodel.input, output = predictions)

model_final.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

model_final.summary()







#es = EarlyStopping(monitor='val_accuracy', mode='max', patience=5,  restore_best_weights=True)
#train_labels_cat = to_categorical(train_labels_np, num_classes= 4)
#test_labels_cat = to_categorical(test_labels_np, num_classes= 4)

#model_final.fit(train_images, train_labels_cat, epochs=10, validation_split=0.2, batch_size=32, callbacks=[es])
#model_final.summary(

history = model_final.fit(train_images_np, train_labels_np, epochs=10,)
#checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
#early = EarlyStopping(monitor='val_acc', min_delta=0, patience=40, verbose=1, mode='auto')
#model_final.fit_generator(generator= traindata, steps_per_epoch= 2, epochs= 100, validation_data= testdata, validation_steps=1, callbacks=[checkpoint,early])
#model_final.fit_generator(generator= train_images_np, steps_per_epoch= 2, epochs= 10)
model_final.save_weights("vgg16_1.h5")

"""### Evaluate the model"""

plt.plot(history.history['accuracy'], label='accuracy')
#plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.savefig("VGGplotTrain.jpg")


test_loss, test_acc = model_final.evaluate(test_images_np,  test_labels_np, verbose=2)
model_final.evaluate(test_images_np,  test_labels_np,)

y_pred = model_final.predict(test_images_np)
y_pred = np.argmax(y_pred, axis=1)
conf_mat = confusion_matrix(test_labels_np, y_pred)
ax= plt.subplot()
sns.heatmap(conf_mat, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['notumor', 'glioma','meningioma','pituitary']); ax.yaxis.set_ticklabels(['notumor', 'glioma','meningioma','pituitary']);
plt.savefig('confusion_matrixVGG16BT.jpg')
print(classification_report(test_labels_np, y_pred, target_names=['notumor', 'glioma','meningioma','pituitary']))





from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D,Input
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
import numpy as np

# Merge inputs and targets
inputs = np.concatenate((train_images_np, test_images_np), axis=0)
targets = np.concatenate((train_labels_np, test_labels_np), axis=0)
acc_per_fold = []
loss_per_fold = []
# Define the K-fold Cross Validator
kfold = KFold(n_splits=5, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(inputs, targets):
    model = models.Sequential()
    model.add(Input(shape = (128,128,3)))
    model.add(vggmodel)
    model.add(Flatten())
    model.add(Dense(50,activation='relu'))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(4,activation = 'softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'])
    print(f'Training for fold {fold_no} ...')
    print('------------------------------------------------------------------------')
    history = model.fit(inputs[train], targets[train],epochs=10)
    model.save_weights("Pesi-VGG16 "+str(fold_no)+"iteration.h5")
    scores = model.evaluate(inputs[test], targets[test], verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    #Stampe e Matrici
    y_pred = model.predict(inputs[test])
    y_pred = np.argmax(y_pred, axis=1)
    conf_mat = confusion_matrix(targets[test],y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = conf_mat, display_labels = ['notumor', 'glioma','meningioma','pituitary'])
    cm_display.plot(cmap='rocket')
    cm_display.figure_.savefig('confusion_matrixVGG_KFold '+str(fold_no)+' iteration.jpg')
    print(classification_report(targets[test], y_pred, target_names=['notumor', 'glioma','meningioma','pituitary']))
    fold_no = fold_no + 1


# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')
