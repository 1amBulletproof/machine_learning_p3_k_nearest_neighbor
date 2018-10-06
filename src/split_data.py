#!/usr/local/bin/python3

#@author		Brandon Tarney
#@date			9/30/2018
#@description	pre_process data


import argparse
from file_manager import FileManager
from data_manipulator import DataManipulator
import numpy as np

#=============================
# split_data
#
#	- read-in data, move class to final col, 
#		remove columns (if necessary) , normalize data, 
#		split into 5 groups for 5 fold cross val & output to file
#=============================
def split_data(file_path, is_random, separator=','):

	#GET DATA
	original_data = FileManager.get_csv_file_data_pandas(file_path, separator)
	#print('original data')
	#print(original_data)

	#STANDARD STUFF

	#Split the data into 5 groups
	groups = list()
	num_groups = 5
	if (is_random == True):
		#Use basic random 5-way split
		groups = DataManipulator.split_data_randomly(original_data, num_groups)
	else:
		#Use more complex split
		groups = DataManipulator.split_data_randomly_accounting_for_class(original_data, num_groups)

	return groups


#=============================
# MAIN PROGRAM
#=============================
def main():
	#print('LOG: Main program to pre-process House-Votes-84.data file')
	parser = argparse.ArgumentParser(description='Pre-process data by splitting it into 5 groups')
	parser.add_argument('file_path', type=str, help='full path to input file')
	parser.add_argument('separator', type=str, help='separator for data')
	parser.add_argument('-o', action='store_true', help='output results to file')
	parser.add_argument('-r', action='store_true', help='totally random groups, may have disproportionate number of a given class')
	args = parser.parse_args()
	print(args)
	file_path = args.file_path
	separator = args.separator
	is_random = args.r
	output_to_file = args.o

	groups = split_data(file_path, is_random, separator)

	for counter, group in enumerate(groups):
		if output_to_file:
			group_file_name = "data_" + str(counter)
			print(group_file_name)
			print(group)
			np.savetxt(group_file_name, group, delimiter=separator)
		print('group: ', counter)
		print('lenth: ', len(group))
		print(group)


if __name__ == '__main__':
	main()
