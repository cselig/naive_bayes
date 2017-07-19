from NaiveBayes import NB



def classify(num_buckets, data, test_data, test_classes):
	model = NB(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], reporting = False, buckets = num_buckets)
	model.train(data)
	predictions = model.classify(test_data)
	correct = 0
	for i in range(0, 30):
	    if predictions[i] == test_classes[i]:
	        correct += 1
	print('num_buckets: ' + str(num_buckets))
	print('Accuracy: ' + str(correct / 30.0))
	print()


iris_data = open('iris_data.txt', 'r')

training_classes = []
training_data = []
test_data = []
test_classes = []

for i, l in enumerate(iris_data):
    split = l.split(',')
    if i % 50 < 40:
        training_classes.append(split[-1].strip())
        training_data.append([float(x) for x in split[:-1]])
    else: 
        test_classes.append(split[-1].strip())
        test_data.append([float(x) for x in split[:-1]])

data = []
for i in range(len(training_data)):
    data.append((training_classes[i], training_data[i]))

for num_buckets in range(1, 11):
	classify(num_buckets, data, test_data, test_classes)




