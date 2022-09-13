import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import cv2
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
data_dir= r'C:\Users\admin\Downloads\Data\train'
val_dir=r'C:\Users\admin\Downloads\Data\val'
def resize(path):
    image_folder = []
    filelist = os.listdir(path)
    for filename in filelist:
        image_folder.append(os.path.join(path, filename))
    height = 0
    width = 0
    length_image = 0
    for j in range(len(image_folder)):
        path = image_folder[j]
        image_path = []
        for i in os.listdir(path):
            path_image = path + f"/{i}"
            image_path.append(path_image)
            image = cv2.imread(path_image)
            height = height + image.shape[0]
            width = width + image.shape[1]
        length_image = length_image + len(image_path)
    return int(height/length_image),int(width/length_image)
height,width=resize(data_dir)
print(resize(data_dir))
batch_size=32
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  image_size=(height, width),
  batch_size=batch_size)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  val_dir,
  image_size=(height, width),
  batch_size=batch_size)
model=Sequential([
    layers.Conv2D(32, 3,activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, 3,activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, 3,padding='same',activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, 3,padding='same'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dropout(0.3),
    layers.Dense(128,activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(5,activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
epochs=20
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
model.save('Classify/model.h5')
model = keras.models.load_model("Classify/model.h5")
class_names = train_ds.class_names
path = r'C:\Users\admin\Downloads\Val_1'
for i in os.listdir(path):
    sunflower_path=(path + f"/{i}")
    img = keras.preprocessing.image.load_img(
        sunflower_path, target_size=(height, width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
