#!/usr/local/bin/python3

#@author		Brandon Tarney
#@date			9/30/2018
#@description	KNearestNeighbor class

import numpy as np
import math as math
import pandas as pd
import argparse
import operator
import random
import copy
from base_model2 import BaseModel

#=============================
# KNearestNeighbor
#
# - Class to encapsulate a K-NN model for testing
#=============================
class KNearestNeighbor(BaseModel) :

	#Assumes data is numpy 2d array of points
	def __init__(self, data, k_number):
		BaseModel.__init__(self, data)
		self.k_number = k_number

	#=============================
	# train()
	#	- Just set data for k-nearest-neighbor - its lazy
	#=============================
	def train(self, new_data):
		self.data = new_data
		return

	#=============================
	# _get_all_data_distances_to_point()
	#	- get all the distances between a given point and a 2d array of points
	#
	#@param	point	point to find distance-from
	#@return	 array of distances between given point and all data
	#=============================
	def _get_all_data_distances_to_point(self, data, point):
		#Remove the classification assumed to be final column
		data_no_class = data[:,:-1]
		#print('data_no_class')
		#print(data_no_class)
		point_no_class = point[:-1] 
		#print('point_no_class')
		#print(point_no_class)
		# sqrt( (X1 - Y1)^2 + (X2 - Y2)^2 .... N )
		square_of_dif = np.square(data_no_class - point_no_class)
		sum_of_dif = np.sum(square_of_dif, axis=1)
		distances = np.sqrt(sum_of_dif)
		return distances

	#=============================
	# _get_closest_points()
	#	- get the classification from the provided points
	#
	#@param	points	points - the closest ones
	#@return	 closest points
	#=============================
	def _get_closest_k_points(self, data, distances, k):
		sorted_distances = distances.argsort()
		#print('sorted_distances')
		#print(sorted_distances)
		k_point_idx = sorted_distances[:k]
		#print('k_point_idx')
		#print(k_point_idx)
		k_points = data[k_point_idx]
		#print('k_points')
		#print(k_points)
		return k_points

	#=============================
	# _get_classification_majority_from_points()
	#	- get the classification from the provided points
	#
	#@param	points	points - the closest ones
	#@return	 classification
	#=============================
	def _get_classification_majority_from_points(self, points):
		#Get the counts of every unique class
		(class_values, class_counts) = np.unique(points[:,-1], return_counts=True)
		#Get the maximum occurences of class and select that class
		majority_class = class_values[np.argmax(class_counts)]
		return majority_class

	#=============================
	# _get_classification_average_from_points()
	#	- get the classification from the provided points
	#
	#@param	points	points - the closest ones
	#@return	 classification
	#=============================
	def _get_classification_average_from_points(self, points):
		#print('closest points')
		#print(points)
		#Get the final column values
		class_values = points[:,-1]
		#print('point class values')
		#print(class_values)
		class_average = np.average(class_values)
		#print('class average')
		#print(class_average)
		return class_average

	#=============================
	# _update_majority_test_results()
	#	- update test performance
	#
	#@param	point	point under consideration
	#@param	classification model calculated classification
	#@return	 
	#=============================
	def _update_majority_test_results(self, point, classification):
		self.tests_completed = self.tests_completed + 1
		test_result = point[-1] == classification
		if (test_result == True):
			self.tests_correct = self.tests_correct + 1
		return test_result

	#=============================
	# _update_average_test_results()
	#	- update test performance
	#
	#@param	point	point under consideration
	#@param	classification model calculated classification
	#@return	 
	#=============================
	def _update_average_test_results(self, point, classification):
		self.tests_completed = self.tests_completed + 1
		#print('average of k points: ', classification, ' actual val: ', point)
		test_result = math.pow(point[-1] - classification, 2) 
		self.tests_square_error = self.tests_square_error + test_result
		return test_result

	#=============================
	# test()
	#	- test test_data via k-nearest neighbor in train_data
	#@return	accuracy as a percentage
	#=============================
	def test(self, test_data, classification_mode):
		#Reset statistics 
		self.tests_completed = 0
		self.tests_correct = 0
		self.tests_square_error = 0

		#print('test_data')
		#print(test_data)
		#Classify every point in test data
		for row in test_data:
			#Remove the classificaiton from distance calc
			#print('row')
			#print(row)
			distances  = self._get_all_data_distances_to_point(self.data, row)
			#print('distances')
			#print(distances)
			#Find the closest k points (neighbors)
			closest_k_points = self._get_closest_k_points(
					self.data,
					distances, 
					self.k_number)
			#print('closest_k_points')
			#print(closest_k_points)
			#Find the classification
			if (classification_mode == "average"):
				#Get average value of closest points - used for regression
				class_average = self._get_classification_average_from_points(closest_k_points)
				#print('class_average')
				#print(class_average)
				#Update overall results
				test_result = self._update_average_test_results(
						row, class_average)
			else:
				#Get majority value of closest points - used for classification
				class_majority = self._get_classification_majority_from_points(closest_k_points)
				#print('classification')
				#print(classification)
				#Update overall results
				test_result = self._update_majority_test_results(
						row, class_majority)

		if (classification_mode == "average"):
			#Mean Squared Error
			result = float(self.tests_square_error / self.tests_completed);
		else:
			#Accuracy = correct / total # tests
			result = 100 * float(self.tests_correct / self.tests_completed)

		return result

		#TODO: need mode or separate method for regression data where you simply take the average of the closest k points

#=============================
# MAIN PROGRAM
#=============================
def main():
	print('Main() - testing knn model')
	parser = argparse.ArgumentParser(description='test knn model')
	parser.add_argument('k_number', type=int, default=1, nargs='?', help='number of total neighbors')
	parser.add_argument('mode', type=str, default="", nargs='?', help='type of classification - "average" or "majority"')
	args = parser.parse_args()

	print()
	print('TEST 1: dummy data')
	print('train data (final col is class):')
	train_data = pd.DataFrame([
		[0,0,0,0], [2,0,0,0], [0,2,0,0], [0,0,2,0],
		[4,4,4,4], [3,4,4,4], [4,3,4,4], [4,4,3,4],
		[8,8,8,8], [6,8,8,8], [8,6,8,8], [8,8,6,8]])
	print(train_data)
	print('test data1 (final col class):')
	test_data1 = pd.DataFrame([[0,0,0,0]])
	print(test_data1)
	print('test data2 (final col class):')
	test_data2 = pd.DataFrame([
		[1,0,3,0], [5,4,3,4], [7,6,7,8]])
	print(test_data2)
	print()

	k_nearest_neighbor = KNearestNeighbor(train_data.values, args.k_number)
	result1 = k_nearest_neighbor.test(test_data1.values, args.mode)
	if (args.mode == "average"):
		print('Total Mean Squared Error:')
		print(result1)
	else:
		print('Result1 Accuracy (%):') 
		print(result1, '%')
	result2 = k_nearest_neighbor.test(test_data2.values, args.mode)
	if (args.mode == "average"):
		print('Total Mean Squared Error:')
		print(result2)
	else:
		print('Result1 Accuracy (%):') 
		print(result2, '%')


if __name__ == '__main__':
	main()
