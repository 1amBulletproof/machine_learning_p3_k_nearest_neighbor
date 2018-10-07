#!/usr/local/bin/python3

#@author		Brandon Tarney
#@date			9/30/2018
#@description	run experiment

from k_nearest_neighbor import KNearestNeighbor
from condensed_k_nearest_neighbor import CondensedKNearestNeighbor

import argparse
from file_manager import FileManager
from data_manipulator import DataManipulator
import numpy as np

#=============================
# run_model()
#
#	- read-in 5 groups of input data, train on 4/5,
#		test on 5th, cycle the 4/5 & repeat 5 times
#		Record overall result!
#=============================
def run_model_with_cross_validation(model_name, knn_mode, k_number):

	#GET DATA
	#- expect data_0 ... data_4
	data_groups = list()
	data_groups.append(np.loadtxt('data_0', delimiter=','))
	data_groups.append(np.loadtxt('data_1', delimiter=','))
	data_groups.append(np.loadtxt('data_2', delimiter=','))
	data_groups.append(np.loadtxt('data_3', delimiter=','))
	data_groups.append(np.loadtxt('data_4', delimiter=','))

	NUM_GROUPS = len(data_groups)

	#For each data_group, train on all others and test on me
	culminating_result = 0;

	for test_group_id in range(NUM_GROUPS):

		#Form training data as 4/5 data
		train_data = np.array([])
		for train_group_id in range(len(data_groups)):
			if (train_group_id != test_group_id):
				#Initialize train_data if necessary
				if (train_data.size == 0):
					train_data = np.copy(data_groups[train_group_id])
				else:
					train_data = np.concatenate(
							(train_data, data_groups[train_group_id]), axis=0)

		print('train_data, group ', str(test_group_id), 'length: ', len(train_data))
		print(train_data)

		test_data = data_groups[test_group_id]

		result = 0
		model = None
		if (model_name == 'knn'):
			model = KNearestNeighbor(train_data, k_number)
			model.train(train_data)
			print('KNN train data length', len(model.data))
			result = model.test(test_data, knn_mode)
		elif (model_name == 'c_knn'):
			model = CondensedKNearestNeighbor(train_data, k_number)
			#Mode is always majority...this is not used for regression
			mode = "majority"
			model.train(train_data)
			print('condensed KNN train data length', len(model.data))
			result = model.test(test_data, mode)
		else:
			print('error - ', model_name, ' is not a supported model')
			return

		print('test_data, group ', str(test_group_id), 'length:', len(test_data))
		print(test_data)

		print()
		print('result of iteration ' + str(test_group_id))
		print(result)
		print()

		culminating_result = culminating_result + result
	
	final_average_result = culminating_result / NUM_GROUPS
	print()
	print('final average result:')
	print(final_average_result)
	print()

	return final_average_result


#=============================
# MAIN PROGRAM
#=============================
def main():
	#print('LOG: Main program to pre-process House-Votes-84.data file')
	parser = argparse.ArgumentParser(description='Pre-process "breast-cancer-wisonsin.data"')
	parser.add_argument('model_name', type=str, help='model name, either "knn" or "c_knn"')
	parser.add_argument('k_number', type=int, help='number of k or neighborhoods')
	parser.add_argument('knn_mode', type=str, default="majority", nargs='?', help='type of classification - "average" or "majority"')
	args = parser.parse_args()
	#print(args)

	final_result = run_model_with_cross_validation(
			args.model_name, args.knn_mode, args.k_number)
	if (args.model_name == 'knn'):
		if (args.knn_mode == 'majority'):
			print('Average Accuracy (%):') 
			print(final_result, '%')
		else:
			print('Average Mean Squared Error:')
			print(final_result)
	else:
		print('Average Accuracy (%):') 
		print(final_result, '%')


if __name__ == '__main__':
	main()
