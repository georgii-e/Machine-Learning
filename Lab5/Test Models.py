import numpy as np
from keras.models import load_model
from PIL import Image

model = load_model(r'F:\Egor\Уроки\Машинне навчання\Лаб5\digits_model.h5')

test_digit = Image.open(r'F:\Egor\Уроки\Машинне навчання\Лаб5\1.png').convert('L')
test_digit = test_digit.resize((28, 28))
test_digit = 1 - np.array(test_digit) / 255
test_digit = test_digit.reshape((1, 28*28))

prediction = list(model.predict(test_digit)[0])
print(prediction.index(max(prediction)))
