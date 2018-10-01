#!/usr/local/bin/python3

#@author		Brandon Tarney
#@date			9/30/2018
#@description	TestModel class

import numpy as np
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
	#	- N/A for k-nearest-neighbor - its lazy
	#=============================
	def train(self):
		return

	#=============================
	# _get_all_data_distances_to_point()
	#	- get all the distances between a given point and a 2d array of points
	#
	#@param	point	point to find distance-from
	#@return	 array of distances between given point and all data
	#=============================
	def _get_all_data_distances_to_point(self, point):
		#Remove the classification assumed to be final column
		data_no_class = self.data[:,:-1]
		point_no_class = point[:-1] 
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
	def _get_closest_points(self, distances):
		sorted_distances = distances.argsort()
		k_point_idx = sorted_distances[:self.k_number]
		k_points = self.data[k_point_idx]
		return k_points

	#=============================
	# _get_classification()
	#	- get the classification from the provided points
	#
	#@param	points	points - the closest ones
	#@return	 classification
	#=============================
	def _get_classification(self, points ):
		#Get the counts of every unique class
		(class_values, class_counts) = np.unique(points[:,-1], return_counts=True)
		#Get the maximum occurences of class and select that class
		winning_class = class_values[np.argmax(class_counts)]
		return winning_class

	#=============================
	# _update_test_results()
	#	- update test performance
	#
	#@param	point	point under consideration
	#@param	classification model calculated classification
	#@return	 
	#=============================
	def _update_test_results(self, point, classification):
		self.tests_completed = self.tests_completed + 1
		test_result = point[-1] == classification
		if (test_result == True):
			self.tests_correct = self.tests_correct + 1
		return test_result

	#=============================
	# test()
	#	- test test_data via k-nearest neighbor in train_data
	#@return	accuracy as a percentage
	#=============================
	def test(self, test_data):
		#Reset statistics 
		self.tests_completed = 0
		self.tests_correct = 0

		print('test_data')
		print(test_data)
		#Classify every point in test data
		for row in test_data:
			#Remove the classificaiton from distance calc
			print('row')
			print(row)
			distances  = self._get_all_data_distances_to_point(row)
			print('distances')
			print(distances)
			#Find the closest k points (neighbors)
			closest_k_points = self._get_closest_points(distances)
			print('closest_k_points')
			print(closest_k_points)
			#Find the classification
			classification = self._get_classification(closest_k_points)
			print('classification')
			print(classification)
			#Update overall results
			test_result = self._update_test_results(
					row, classification)

		#Accuracy = correct / total # tests
		percent_correct = 100 * float(self.tests_correct / self.tests_completed)
		return percent_correct


#=============================
# MAIN PROGRAM
#=============================
def main():
	print('Main() - testing knn model')
	parser = argparse.ArgumentParser(description='test knn model')
	parser.add_argument('k_number', type=int, default=1, nargs='?', help='number of total neighbors')
	args = parser.parse_args()

	print()
	print('TEST 1: dummy data')
	print('train data (final col is class):')
	train_data = pd.DataFrame([
		[0,0,0,0], [1,0,0,0], [0,1,0,0], [0,0,1,0],
		[1,1,1,1], [0,1,1,1], [1,0,1,1], [1,1,0,1],
		[2,2,2,2], [1,2,2,2], [2,1,2,2], [2,2,1,2]])
	print(train_data)
	print('test data1 (final col class):')
	test_data1 = pd.DataFrame([[0,0,0,0]])
	print(test_data1)
	print('test data2 (final col class):')
	test_data2 = pd.DataFrame([
		[0,1,1,1], [1,2,1,1], [2,0,2,2]])
	print(test_data2)
	print()

	k_nearest_neighbor = KNearestNeighbor(train_data.values, args.k_number)
	result1 = k_nearest_neighbor.test(test_data1.values)
	print('Result1 Accuracy (%):') 
	print(result1, '%')
	result2 = k_nearest_neighbor.test(test_data2.values)
	print('Result2 Accuracy (%):') 
	print(result2, '%')


if __name__ == '__main__':
	main()
