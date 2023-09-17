import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import random

def brightness_augment(img, factor=0.5):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) #convert to hsv
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 2] = hsv[:, :, 2] * (np.random.uniform(0.8,1.2)) #scale channel V uniformly
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255 #reset out of range values
    rgb = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2RGB)
    return rgb


def sharp(img):
    kernel = np.array([[0, -1, 0],
                 [-1, 5,-1],
                 [0, -1, 0]])
    image_sharp = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    return image_sharp

def rotate(img):
    from scipy import ndimage
    import random
    rand = random.randint(0,360)
    #rotation angle in degree
    rotated = ndimage.rotate(img, rand)
    return rotated



def carica_imgs(dataset):
  images = []
  for img_path in dataset:
       img0 = cv2.imread(img_path)
       img0 = cv2.resize(img0, (128,128), interpolation = cv2.INTER_AREA)
       #img0 = rotate(img0)
       img0 = sharp(img0)
       img0 = brightness_augment(img0)
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

train_images = carica_imgs(train_path)
test_images = carica_imgs(test_path)
train_labels_np = np.array(train_labels)
test_images_np = np.array(test_images)
test_labels_np = np.array(test_labels)
train_images_np = np.array(train_images)
train_labels_np = train_labels_np.reshape(len(train_labels_np),1)
test_labels_np = test_labels_np.reshape(len(test_labels_np),1)


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(4,activation = "softmax"))
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])


#history = model.fit(train_images_np, train_labels_np, epochs=10, validation_data=(test_images_np, test_labels_np))
history = model.fit(train_images_np, train_labels_np, epochs=10,)


path ="C:/Users/ilcai/Desktop/ProgettoML/Pesi/Dataset/"
#model.load_weights(path+"cnnDatasetPulito.h5")
model.save_weights(path+"cnnDatasetPulito.h5")

"""### Evaluate the model"""

plt.plot(history.history['accuracy'], label='accuracy')
#plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images_np,  test_labels_np, verbose=2)
plt.savefig("BTplotTrain.jpg")
y_pred = model.predict(test_images_np)
y_pred = np.argmax(y_pred, axis=1)
conf_mat = confusion_matrix(test_labels_np, y_pred)
ax= plt.subplot()
sns.heatmap(conf_mat, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['notumor', 'glioma','meningioma','pituitary']); ax.yaxis.set_ticklabels(['notumor', 'glioma','meningioma','pituitary']);
plt.savefig('confusion_matrixCNN_DATASETPULITO.jpg')

from sklearn.metrics import classification_report
print(classification_report(test_labels_np, y_pred, target_names=['notumor', 'glioma','meningioma','pituitary']))




from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
import numpy as np

# Merge inputs and targets
inputs = np.concatenate((train_images_np, test_images_np), axis=0)
targets = np.concatenate((train_labels_np, test_labels_np), axis=0)
acc_per_fold = []
loss_per_fold = []
conf_mat_fold = []
# Define the K-fold Cross Validator
kfold = KFold(n_splits=5, shuffle=True)

# K-fold Cross Validation model

fold_no = 1
for train, test in kfold.split(inputs, targets):
 model = models.Sequential()
 model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
 model.add(layers.MaxPooling2D((2, 2)))
 model.add(layers.Conv2D(64, (3, 3), activation='relu'))
 model.add(layers.MaxPooling2D((2, 2)))
 model.add(layers.Conv2D(64, (3, 3), activation='relu'))
 model.add(layers.Flatten())
 model.add(layers.Dense(64, activation='relu'))
 model.add(layers.Dense(4,activation = "softmax"))
 model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
 print('------------------------------------------------------------------------')
 print(f'Training for fold {fold_no} ...')
 history = model.fit(inputs[train], targets[train],epochs=10)
 model.save_weights("Pesi-CNN "+str(fold_no)+"iteration.h5")
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
 cm_display.figure_.savefig('confusion_matrixKFoldCNN_DATASETPULITO '+str(fold_no)+' iteration.jpg')
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
