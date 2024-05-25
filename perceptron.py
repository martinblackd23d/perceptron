import random
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

class perceptron:
	lr = 0.01
	epochs = 100
	weights = None
	bias = None

	def __init__(self, lr=0.01, n_epochs=100, n_features=2):
		self.lr = lr
		self.epochs = n_epochs
		self.weights = []
		for i in range(n_features + 1):
			self.weights.append(random.uniform(-1, 1))
		self.bias = -1

	def predict(self, x):
		x = [self.bias] + list(x)
		s = 0
		for i in range(len(x)):
			s += x[i] * self.weights[i]
		if s > 0:
			return 1
		else:
			return 0
		
	def fit(self, data, target):
		convergence = []
		for epoch in range(self.epochs):
			accuracte_predictions = 0
			print(f'Epoch {epoch}')
			for i in range(len(data)):
				prediction = self.predict(data[i])
				x = [self.bias] + list(data[i])
				if prediction == target[i]:
					accuracte_predictions += 1

				for j in range(len(self.weights)):
					self.weights[j] -= self.lr * (prediction - target[i]) * x[j]
			convergence.append(accuracte_predictions / len(data))

def main():
	data, targets = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_classes = 2, n_clusters_per_class=1)
	data_train, data_test, target_train, target_test = train_test_split(data, targets, test_size=0.3)
	perceptron = perceptron()

	perceptron.fit(data_train, target_train)
	predictions = []
	for i in range(len(data_test)):
		predictions.append(perceptron.predict(data_test[i]))
	print("accuracy:")
	print(accuracy_score(target_test, predictions))

	# training data
	plt.scatter(data_train[:, 0], data_train[:, 1], c=target_train)
	print(perceptron.weights)
	slope = - 1 * perceptron.weights[1] / perceptron.weights[2]
	intercept = perceptron.weights[0] / perceptron.weights[2]
	x = np.linspace(-5, 5, 100)
	y = slope * x + intercept
	plt.plot(x, y)
	plt.show()


	# testing data
	plt.scatter(data_test[:, 0], data_test[:, 1], c=target_test)
	print(perceptron.weights)
	slope = - 1 * perceptron.weights[1] / perceptron.weights[2]
	intercept = perceptron.weights[0] / perceptron.weights[2]
	x = np.linspace(-5, 5, 100)
	y = slope * x + intercept
	plt.plot(x, y)
	plt.show()

if __name__ == '__main__':
	main()



