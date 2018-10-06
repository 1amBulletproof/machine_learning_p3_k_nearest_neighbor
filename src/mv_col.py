
	#Move class column to final col
	data = original_data
	print(class_column)
	if (class_column != -1):
		data_as_np = DataManipulator.move_column_to_end(original_data.values, class_column)
		data = pd.DataFrame(data_as_np)
	print('data')
	print(data_as_np)
