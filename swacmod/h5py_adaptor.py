import h5py

def write_h5py(file_path, dataset_name, data):
    with h5py.File(file_path, "w") as outfile:
        outfile.create_dataset(dataset_name, data=data, compression="gzip")
