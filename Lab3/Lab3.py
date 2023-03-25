import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap

pd.set_option('display.precision', 2)
plt.style.use('seaborn-v0_8')

initial_df = pd.read_csv(r'F:\Egor\Уроки\Машинне навчання\Лаб2\1895-2018.csv')
nyc = initial_df.copy()
nyc.columns = ['Date', 'Temperature', 'Anomaly']
nyc.drop(index=range(4), inplace=True)
nyc.reset_index(drop=True, inplace=True)
nyc = nyc.astype({'Date': 'int64', 'Temperature': 'float'})
nyc.loc[:, 'Date'] = nyc.loc[:, 'Date'].floordiv(100)
dates = nyc.loc[:, 'Date'].values
temps = nyc.loc[:, 'Temperature'].values

X_train, X_test, Y_train, Y_test = train_test_split(nyc.loc[:, 'Date'].values.reshape(-1, 1),
                                                    nyc.loc[:, 'Temperature'].values, random_state=11)
linear_regression = LinearRegression()
linear_regression.fit(X=X_train, y=Y_train)
k = linear_regression.coef_
b = linear_regression.intercept_
predicted = linear_regression.predict(X_test)
expected = Y_test
for p, e in zip(predicted[::5], expected[::5]):
    print(f'predicted: {p:.2f}, expected: {e:.2f}, difference: {(e-p)/p*100:.2f}%')
predict = (lambda x: linear_regression.coef_ * x + linear_regression.intercept_)
print('Predicted temperature in 2019:', predict(2019))
print('Predicted temperature in 1890:', predict(1890))
axes = sns.scatterplot(data=nyc, x='Date', y='Temperature', hue='Temperature', palette='winter', legend=False)
axes.set_ylim(10, 70)
x = np.array([min(nyc.loc[:, 'Date'].values), max(nyc.loc[:, 'Date'].values)])
y = predict(x)
line = plt.plot(x, y)
plt.show()
# part 2
np.random.seed(1)
X_xor = np.random.randn(200, 2)
Y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
Y_xor = np.where(Y_xor, 1, -1)
plt.scatter(X_xor[Y_xor == 1, 0], X_xor[Y_xor == 1, 1], c='b', marker='x', label='1')
plt.scatter(X_xor[Y_xor == -1, 0], X_xor[Y_xor == -1, 1], c='r', marker='s', label='-1')
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.legend(loc='best')
plt.show()


def plot_decision_regions(X, Y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(Y))])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, c1 in enumerate(np.unique(Y)):
        plt.scatter(x=X[Y == c1, 0], y=X[Y == c1, 1], alpha=0.8, c=colors[idx], marker=markers[idx], label=c1)

    if test_idx:
        X_test, Y_test = X[test_idx, :], Y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', edgecolor='black', alpha=1.0, linewidth=1, marker='0', s=100)


svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
svm.fit(X_xor, Y_xor)
plot_decision_regions(X_xor, Y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.show()

# part 3

iris = load_iris()

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=11)

# Initialize classifiers
knn = KNeighborsClassifier()
svc = SVC()
gnb = GaussianNB()

# Train classifiers
knn.fit(X_train, y_train)
svc.fit(X_train, y_train)
gnb.fit(X_train, y_train)

# Make predictions on testing data
knn_prediction = knn.predict(X_test)
svc_prediction = svc.predict(X_test)
gnb_prediction = gnb.predict(X_test)

# Calculate accuracy scores
knn_accuracy = accuracy_score(y_test, knn_prediction)
svc_accuracy = accuracy_score(y_test, svc_prediction)
gnb_accuracy = accuracy_score(y_test, gnb_prediction)

# Print accuracy scores
print(f'KNN Accuracy: {knn_accuracy:.3f}')
print(f'SVC Accuracy: {svc_accuracy:.3f}')
print(f'GaussianNB Accuracy: {gnb_accuracy:.3f}')

# Plot results
labels = ['KNN', 'SVC', 'GaussianNB']
accuracies = [knn_accuracy, svc_accuracy, gnb_accuracy]
plt.figure(figsize=(8, 6))
plt.bar(labels, accuracies)
plt.title('Classifier Comparison on Iris Dataset')
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.ylim(0.85, 1)
plt.show()
