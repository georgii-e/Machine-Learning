import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier

pd.set_option('display.precision', 2)
plt.style.use('seaborn-v0_8')

digits = load_digits()


def show_digits(row_amount, col_amount):
    figure, axes = plt.subplots(nrows=row_amount, ncols=col_amount, figsize=(col_amount, row_amount))
    for item in zip(axes.ravel(), digits.images, digits.target):
        axes, image, target = item
        axes.imshow(image)
        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_title(target)
    plt.tight_layout()
    plt.show()


show_digits(4, 6)
show_digits(6, 6)

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state=11, train_size=0.8)
print('X_test size: ', X_test.shape)
print('X_train size: ', X_train.shape)

knn = KNeighborsClassifier(n_neighbors=5)
svc = SVC()
gnb = GaussianNB()

knn.fit(X_train, y_train)
svc.fit(X_train, y_train)
gnb.fit(X_train, y_train)

knn_predicted = knn.predict(X_test)
expected = y_test
print('predicted: ', knn_predicted[:36])
print('expected:  ', expected[:36])

print(f'KNN Accuracy: {knn.score(X_test, y_test) * 100:.3f}%')
print(f'SVC Accuracy: {svc.score(X_test, y_test) * 100:.3f}%')
print(f'GNB Accuracy: {gnb.score(X_test, y_test) * 100:.3f}%')

confusion = confusion_matrix(y_true=expected, y_pred=knn_predicted)
print(confusion)

names = [str(digit) for digit in digits.target_names]
print(classification_report(expected, knn_predicted, target_names=names))

k_range = range(1, 15)
accuracy_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)  # Train the classifier using the current value of k
    accuracy = knn.score(X_test, y_test)  # Evaluate the accuracy of the classifier on the test set
    accuracy_scores.append(accuracy)  # Append the accuracy score to the list

# Generate a plot of the accuracy scores as a function of k
plt.plot(k_range, accuracy_scores)
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('KNN Classifier Accuracy')
plt.show()
print(
    f'Best accuracy: {max(accuracy_scores) * 100:.2f} with {accuracy_scores.index(max(accuracy_scores)) + 1} neighbours')
