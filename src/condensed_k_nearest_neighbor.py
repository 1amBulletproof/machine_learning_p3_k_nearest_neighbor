#!/usr/local/bin/python3

#@author		Brandon Tarney
#@date			9/30/2018
#@description	CondensedKNearestNeighbor class

import numpy as np
import pandas as pd
import argparse
import operator
import random
import copy
from k_nearest_neighbor import KNearestNeighbor

#=============================
# CondensedKNearestNeighbor
#
# - Class to encapsulate a condensed K-NN model for testing
#=============================
class CondensedKNearestNeighbor(KNearestNeighbor) :

	#Assumes data is numpy 2d array of points
	def __init__(self, data, k_number):
		KNearestNeighbor.__init__(self, data, k_number)

	#=============================
	# train()
	#	- Condense data into subset representing @ least every class
	#=============================
	def train(self, new_data):
		#print('data before condense')
		#print(self.data)
		#Initialize with an empty set
		final_chosen_points = np.array([])

		shuffled_data = np.array(new_data, copy=True)

		#For each point Xt point in new_data, find point X' where
		#	D(Xt, X') = min for all X' in chosen points
		while True:
			#Randomize your data
			np.random.shuffle(shuffled_data)
			#print('shuffled data')
			#print(shuffled_data)

			#initialize values to empty for each run
			#iterative_chosen_points = np.array([])
			iterative_chosen_idx = []

			#randomly grab a starting point
			if final_chosen_points.size == 0:
				#print('first row shuffled data')
				#print(shuffled_data[0])
				final_chosen_points = np.array([shuffled_data[0]])
				#print('final_chosen_points')
				#print(final_chosen_points)

			for point_idx in range(len(shuffled_data)):
				#print('point under consideration')
				#print(shuffled_data[point_idx])
				#find closest point in Z X' to 'point'
				point = shuffled_data[point_idx]
				distances_to_point = self._get_all_data_distances_to_point(
						final_chosen_points, point)
				#print('distances_to_point')
				#print(distances_to_point)
				num_close_points = 1
				closest_points = self._get_closest_k_points(
						final_chosen_points, 
						distances_to_point, 
						num_close_points)

				#if class of 'point' != class of X', add point to Z
				closest_point_class = closest_points[0,-1]
				point_class = point[-1]
				if closest_point_class != point_class:
					#print('chosen points')
					#print(final_chosen_points)
					#print('point in consideration')
					#print(point)
					#print('closest point in final_chosen_points')
					#print(closest_points)
					iterative_chosen_idx.append(point_idx)
					final_chosen_points = np.vstack(
							[final_chosen_points, point])

				#End of for loop

			if (len(iterative_chosen_idx) == 0):
				break # chosen set will be unchanged

			#print('final_chosen_points')
			#print(final_chosen_points)
			break

			#End of while loop

		self.data = np.copy(final_chosen_points)
		#print('data after condense')
		#print(self.data)
		return


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
		[0,0,0,0], [2,0,0,0], [0,2,0,0], [0,0,2,0],
		[4,4,4,4], [3,4,4,4], [4,3,4,4], [4,4,3,4],
		[8,8,8,8], [6,8,8,8], [8,6,8,8], [8,8,6,8]])
	print(train_data)
	print('test data1 (final col class):')
	test_data1 = pd.DataFrame([[0,0,0,0]])
	print(test_data1)
	print('test data2 (final col class):')
	test_data2 = pd.DataFrame([
		[2,4,4,4], [4,6,4,4], [8,2,8,8]])
	print(test_data2)
	print()

	condensed_knn = CondensedKNearestNeighbor(train_data.values, args.k_number)
	condensed_knn.train(train_data.values)
	print('condensed train data')
	print(condensed_knn.data)
	result1 = condensed_knn.test(test_data1.values)
	print('Result1 Accuracy (%):') 
	print(result1, '%')
	result2 = condensed_knn.test(test_data2.values)
	print('Result2 Accuracy (%):') 
	print(result2, '%')


if __name__ == '__main__':
	main()
