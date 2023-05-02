import keras
from keras import layers
from keras.datasets import cifar10
from keras.utils import to_categorical
import matplotlib.pyplot as plt

(train_imgs, train_labels), (test_imgs, test_labels) = cifar10.load_data()
objects = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_imgs[i])
    plt.xlabel(objects[train_labels[i][0]])
plt.show()

train_imgs = train_imgs.astype('float32') / 255
test_imgs = test_imgs.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network = keras.models.Sequential()
network.add(layers.Conv2D(filters=32, kernel_size=(4, 4), input_shape=(32, 32, 3), activation='relu'))
network.add(layers.MaxPool2D(pool_size=(2, 2)))
network.add(layers.Dropout(0.25))  # Drop 25% of the units from the layer.
network.add(layers.Conv2D(filters=32, kernel_size=(4, 4), input_shape=(32, 32, 3), activation='relu'))
network.add(layers.MaxPool2D(pool_size=(2, 2)))
network.add(layers.Dropout(0.25))
network.add(layers.Flatten())
network.add(layers.Dense(256, activation='relu'))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
network.fit(train_imgs, train_labels, epochs=15, batch_size=64)

test_loss, test_acc = network.evaluate(test_imgs, test_labels)

network.save(r'F:\Egor\Уроки\Машинне навчання\Лаб5\cifar10_model.h5')

