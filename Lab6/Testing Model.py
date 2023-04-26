import random
import numpy as np
from keras.models import load_model

model = load_model(r'F:\Egor\Уроки\Машинне навчання\Лаб6\university_model.h5')
math_score = random.randint(100, 200)
english_score = random.randint(100, 200)
ukrainian_score = random.randint(100, 200)
is_privilege = random.choices([0, 1], weights=[0.75, 0.25])[0]
rating = math_score * 0.4 + english_score * 0.3 + ukrainian_score * 0.3

print(f'Math score: {math_score}, English score: {english_score},'
      f' Ukrainian score: {ukrainian_score}, rating: {rating:.1f}, privilege: {is_privilege}')

student = [math_score, english_score, ukrainian_score, rating, is_privilege]
prediction = model.predict(np.array([student]))[0][0]
if prediction < 0.5:
    print(f'Student will not be able to enter a university, predicted value: {prediction:.3f}')
else:
    print(f'Student will be able to enter a university, predicted value: {prediction:.3f}')
