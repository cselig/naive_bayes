import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from NaiveBayes import NB


def classify_iris():
    iris_data = open('iris_data.txt', 'r')

    training_classes = []
    training_data = []
    test_data = []
    test_classes = []

    # process the data into training and testing sets
    for i, l in enumerate(iris_data):
        split = l.split(',')
        # add data to training set
        if i % 50 < 40:
            training_classes.append(split[-1].strip())
            training_data.append([float(x) for x in split[:-1]])
        # add data to test set
        else: 
            test_classes.append(split[-1].strip())
            test_data.append([float(x) for x in split[:-1]])

    data = []
    for i in range(len(training_data)):
        data.append((training_classes[i], training_data[i]))

    # initialize and train model
    model = NB(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
    model.train(data)
    # get predictions from the classifier
    predictions = model.classify(test_data)
    # 
    correct = 0
    for i in range(0, len(test_classes)):
        if predictions[i] == test_classes[i]:
            correct += 1
    print('Accuracy: ' + '%.2f' % (correct / 30.0 * 100) + '%')

if __name__ == '__main__':
	classify_iris()