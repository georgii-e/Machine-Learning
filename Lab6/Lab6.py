import random
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

AMOUNT_OF_APPLICANTS = 6000
MAX_ADMISSIBLE_STUDENTS = int((AMOUNT_OF_APPLICANTS * 70) / 300)  # 1400
MAX_ADMISSIBLE_PRIVILEGE_STUDENTS = int(0.1 * MAX_ADMISSIBLE_STUDENTS)  # 140
MAX_ADMISSIBLE_UNPRIVILEGE_STUDENTS = MAX_ADMISSIBLE_STUDENTS - MAX_ADMISSIBLE_PRIVILEGE_STUDENTS  # 1260


def generate_applicants():
    students = pd.DataFrame(columns=['English', 'Math', 'Ukrainian', 'Rating', 'Privilege', 'Enrolled'])
    for i in range(AMOUNT_OF_APPLICANTS):
        math_score = random.randint(100, 200)
        english_score = random.randint(100, 200)
        ukrainian_score = random.randint(100, 200)
        is_privilege = random.choices([0, 1], weights=[0.75, 0.25])[0]
        student = pd.DataFrame({'English': [english_score], 'Math': [math_score], 'Ukrainian': [ukrainian_score],
                                'Rating': [math_score * 0.4 + english_score * 0.3 + ukrainian_score * 0.3],
                                'Privilege': [is_privilege], 'Enrolled': [1]})
        students = pd.concat([students, student], ignore_index=True)
    return students


def cannot_apply(students):
    students.loc[(students.loc[:, 'Privilege'] == 0) & (students.loc[:, 'Rating'] < 160), 'Enrolled'] = 0
    students.loc[(students.loc[:, 'Privilege'] == 0) & (students.loc[:, 'Math'] < 140), 'Enrolled'] = 0
    students.loc[students.loc[:, 'English'] < 120, 'Enrolled'] = 0
    students.loc[students.loc[:, 'Math'] < 120, 'Enrolled'] = 0
    students.loc[students.loc[:, 'Ukrainian'] < 120, 'Enrolled'] = 0
    students.loc[(students.loc[:, 'Privilege'] == 1) & (students.loc[:, 'Rating'] < 144), 'Enrolled'] = 0
    students.sort_values(by=['Enrolled', 'Rating'], ascending=False, inplace=True)
    students.reset_index(drop=True, inplace=True)
    students.loc[:MAX_ADMISSIBLE_UNPRIVILEGE_STUDENTS, 'Privilege'] = 0  # privilege don't make any sense
    students.loc[(students['Privilege'] == 0) & (students.index >= MAX_ADMISSIBLE_UNPRIVILEGE_STUDENTS), 'Enrolled'] = 0
    students.sort_values(by=['Privilege', 'Enrolled', 'Rating'], ascending=False, inplace=True)
    students.reset_index(drop=True, inplace=True)
    students.loc[(students['Privilege'] == 1) & (students.index >= MAX_ADMISSIBLE_PRIVILEGE_STUDENTS), 'Enrolled'] = 0
    students.sort_values(by=['Enrolled', 'Rating'], ascending=False, inplace=True)
    students.reset_index(drop=True, inplace=True)
    # print(len(students[students.loc[:, 'Enrolled'] == 1]))
    # print(len(students[(students.loc[:, 'Enrolled'] == 1) & (students.loc[:, 'Privilege'] == 1)]))
    return students


def build_model(X_train, y_train):
    model = Sequential()
    model.add(Dense(16, input_dim=5, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=200, batch_size=8)
    model.save(r'F:\Egor\Уроки\Машинне навчання\Лаб6\university_model.h5')
    return model


applicants = generate_applicants()
applicants = cannot_apply(applicants)
train_data, test_data = train_test_split(applicants)

X_train = train_data.drop('Enrolled', axis=1).values.astype('float32')
y_train = train_data['Enrolled'].values.astype('float32')
X_test = test_data.drop('Enrolled', axis=1).values.astype('float32')
y_test = test_data['Enrolled'].values.astype('float32')

network = build_model(X_train, y_train)
# network = load_model(r'F:\Egor\Уроки\Машинне навчання\Лаб6\university_model.h5')
test_loss, test_acc = network.evaluate(X_test, y_test)

y_values = network.predict(X_test)
y_predicted = [1 if y >= 0.5 else 0 for y in y_values]
accuracy = sum(y_predicted == y_test) / len(y_test)
print(f'Test accuracy: {accuracy * 100:.2f}%')
print(len(y_test))

result_df = test_data.assign(predicted_enrollment=y_predicted)
result_df = result_df.assign(predict_confidance=y_values)
result_df.sort_values(by=['predicted_enrollment', 'Enrolled'], inplace=True)
result_df.to_excel(r'F:\Egor\Уроки\Машинне навчання\Лаб6\students.xlsx', index=False)

