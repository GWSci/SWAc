import h5py
import numpy as np

def write_h5py(file_path, dataset_name, data):
    with h5py.File(file_path, "w") as outfile:
        outfile.create_dataset(dataset_name, data=data, compression="gzip")

def read_h5py(path):
	result = {}
	with h5py.File(path, "r") as in_file:
		for dataset_name in in_file.keys():
			dataset = in_file[dataset_name]
			result[dataset_name] = np.array(dataset)
	return result
