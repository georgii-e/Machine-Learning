import keras
from keras import layers
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt

(train_imgs, train_labels), (test_imgs, test_labels) = fashion_mnist.load_data()
objects = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(255 - train_imgs[i], cmap='gray')
    plt.xlabel(objects[train_labels[i]])
plt.show()

train_imgs = train_imgs.reshape((60000, 28 * 28))
train_imgs = train_imgs.astype('float32') / 255
test_imgs = test_imgs.reshape((10000, 28 * 28))
test_imgs = test_imgs.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network = keras.models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(128, activation='relu'))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
network.fit(train_imgs, train_labels, epochs=10, batch_size=256)

test_loss, test_acc = network.evaluate(test_imgs, test_labels)

network.save(r'F:\Egor\Уроки\Машинне навчання\Лаб5\fashion_model.h5')
