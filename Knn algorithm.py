import csv         
import math
import operator
# calculate the similarity in order to make predictions. Use euclidean distance measure
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)
#find k most similar instances (neighbors)
def getNeighbors(loadtrainDataset, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(loadtrainDataset)):
		dist = euclideanDistance(testInstance, loadtrainDataset[x], length)
		distances.append((loadtrainDataset[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors
#devise predicted response based on neighbors
def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]
#classification accuracy, ratio of total correct predictions out of all predictions
def getAccuracy(loadtestDataset, predictions):
	correct = 0
	for x in range(len(loadtestDataset)):
		if loadtestDataset[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(loadtestDataset))) * 100.0
    
	
#main
def main():
    dataSet1 = []                                               
    dataSet2 = []
    loadtrainDataset = []
    loadtestDataset = []
	# handle the data. Read the trainig.dat and testing.dat file and convert it to csv 
	#and also to load test and train data set
    with open('training.dat.csv','rb') as csvfile:          
		lines = csv.reader(csvfile)
		dataSet1 = list(lines)
		for x in range(len(dataSet1)):                        
			for y in range(4):
				dataSet1[x][y] = float(dataSet1[x][y])
				loadtrainDataset.append(dataSet1[x])               
    with open('testing.dat.csv','rb') as csvfile:
		lines = csv.reader(csvfile)
		dataSet2 = list(lines)                             
		for x in range(len(dataSet2)):
			for y in range(4):
			    dataSet2[x][y] = float(dataSet2[x][y])
			    loadtestDataset.append(dataSet2[x])
	#Display the train and test data			
    print 'Train set: ' + repr(len(loadtrainDataset))
    print 'Test set: ' + repr(len(loadtestDataset))
	# generate predictions
    predictions=[]
    k = 3
    for x in range(len(loadtestDataset)):
		neighbors = getNeighbors(loadtrainDataset, loadtestDataset[x], k)
		result = getResponse(neighbors)
		predictions.append(result)
		print('> predicted=' + repr(result) + ', actual=' + repr(loadtestDataset[x][-1]))
    accuracy = getAccuracy(loadtestDataset, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')
    print len(loadtestDataset)
main()
