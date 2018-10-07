#!/usr/local/bin/python3

#@author		Brandon Tarney
#@date			8/31/2018
#@description	Script to normalize the data

from file_manager import FileManager
from data_manipulator import DataManipulator
import csv
import argparse
import numpy as np
from sklearn import preprocessing

#=============================
# MAIN PROGRAM
#=============================
def main():
	#Move class column to final col

	parser = argparse.ArgumentParser(description='Normalize the data')
	parser.add_argument('file_path_in', type=str, help='full path to input file')
	parser.add_argument('file_path_out', type=str, help='full path to output file')
	args = parser.parse_args()

	#read in data, normalize, save to file
	data = FileManager.get_csv_file_data_numpy(args.file_path_in)
	data = data.astype(np.float)
	min_max_scaler = preprocessing.MinMaxScaler()
	normalized_data = min_max_scaler.fit_transform(data)
	print(normalized_data)
	np.savetxt(args.file_path_out, normalized_data, delimiter=',')



if __name__ == '__main__':
	main()
