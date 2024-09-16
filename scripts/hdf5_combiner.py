from pathlib import Path
from tqdm import tqdm
import numpy as np
from glob import glob
from os.path import join
import h5py

class HDF5Combiner:
    def __init__(self, src_paths, out_dir, max_file_size=2000):
        self.src_paths = src_paths
        self.out_dir = out_dir
        self.max_file_size = max_file_size  # in MB
        self.new_file_idx = 0
        self.current_file_size = 0  # in MB
        self.new_file = None
        
        self.data_idx = 0
                
        self.shape = self.get_shape()
                
        # Estimate the number of windows based on shape, data type and max file size
        self.n_windows = int(self.max_file_size * 1e6 / (np.prod(self.shape) * np.dtype(np.float32).itemsize))
        
        
    def get_shape(self):
        with h5py.File(self.src_paths[0], 'r') as file:
            return file['data'].shape[1:]
        
    def update_files_and_idxs(self, files, idxs):
        
        idxs_to_remove = [i for i in range(len(files)) if i not in idxs]
        # Reverse sort the idxs_to_remove to ensure we remove from end to start,
        # which prevents index shift issues during removal.
        for idx in sorted(idxs_to_remove, reverse=True):
            del files[idx]

        # Create a dictionary to map old indexes to new ones
        # Since files at positions in idxs_to_remove are removed, the indexes after those
        # need to be decremented by the number of removals up to that point.
        index_map = {}
        removals_count = 0
        for i in range(len(files) + len(idxs_to_remove)): # original length of files before removal
            if i in idxs_to_remove:
                removals_count += 1
            else:
                index_map[i] = i - removals_count

        # Update idxs based on the new index mapping
        updated_idxs = [index_map[idx] for idx in idxs if idx in index_map] # Ensure idx is in the new mapping
        
        return np.array(files, dtype=h5py.string_dtype()), np.array(updated_idxs, dtype=np.int32)

    def create_new_file(self):
        if self.new_file is not None:
            self.new_file.close()
            
        #file_path = join(self.out_dir, f'combined_{self.new_file_idx}.hdf5')
        # make file_path be combined_00001.hdf5, combined_00002.hdf5, etc.
        file_path = join(self.out_dir, f'combined_{self.new_file_idx:05d}.hdf5')
        
        if Path(file_path).exists():
            raise ValueError(f"File {file_path} already exists. Please delete it and try again.")
        else:
            self.new_file = h5py.File(file_path, 'w')
        
        # Add checksum Fletcher32 
        self.new_file.create_dataset(
            "data", shape=(self.n_windows, *self.shape), chunks=(1, *self.shape),
            maxshape=(None, *self.shape), dtype=np.float32, fletcher32=True
            )
        
        self.new_file_idx += 1
        
        #self.data = np.empty((0, *self.shape), dtype=np.float32)
        self.file_idxs = np.empty((0), dtype=np.int32)
        self.files = np.empty((0), dtype=h5py.string_dtype())
        self.time_slices = np.empty((0, 2), dtype=np.float32)
        self.current_file_size = 0
        self.data_idx = 0

    def save_and_reset(self):        
        
        # Resize the dataset to the correct size
        self.new_file['data'].resize((self.data_idx, *self.shape))
        
        self.files, self.file_idxs = self.update_files_and_idxs(self.files.tolist(), self.file_idxs.tolist())
             
        self.new_file.create_dataset("file_idxs", data=self.file_idxs, dtype=np.int32, fletcher32=True)
        # self.new_file.attrs['file_idxs'] = self.file_idxs
        
        # Save time slices as a dataset
        self.new_file.create_dataset("time_slices", data=self.time_slices, dtype=np.float32, fletcher32=True)
        # self.new_file.attrs['files'] = self.files
        
        self.new_file.attrs['time_slices'] = self.time_slices
        
        assert len(self.time_slices) == len(self.file_idxs) == self.data_idx

    def add_data(self, data, file_idxs, files, time_slices):
        data_size = data.nbytes / 1e6  # Convert bytes to MB
        if self.current_file_size + data_size > self.max_file_size:
            self.save_and_reset()
            self.create_new_file()
    
        current_size = self.new_file['data'].shape[0]
        n_elem = data.shape[0]
        
        # Resize the dataset while self.data_idx+n_elem is out of bounds
        while self.data_idx+data.shape[0] > current_size:
            new_size = current_size + self.n_windows
            self.new_file['data'].resize((new_size, *self.shape))
            current_size = self.new_file['data'].shape[0]
            print(f"Resized dataset to {new_size}")
        
        # Add the data
        self.new_file['data'][self.data_idx:self.data_idx+n_elem] = data
        self.data_idx += n_elem
        
        # Fix the indices
        if len(self.file_idxs) > 0:
            file_idxs = file_idxs + len(self.files) 
                
        if len(file_idxs) > 0:
            self.file_idxs = np.concatenate([self.file_idxs, file_idxs], axis=0)
            
        self.files = np.concatenate([self.files, files], axis=0)
        
        if len(time_slices) > 0:
            self.time_slices = np.concatenate([self.time_slices, time_slices], axis=0)    
        
        
        self.current_file_size += data_size

    def combine(self):
        self.create_new_file()  # Prepare the first new file
        
        for path in tqdm(self.src_paths):
            with h5py.File(path, 'r') as file:
                # Extract data and attributes
                data = file['data'][:]
                file_idxs = file.attrs['file_idxs']
                files = file.attrs['files']
                time_slices = file.attrs['time_slices']

                self.add_data(data, file_idxs, files, time_slices)

        # Close the last file properly
        if self.new_file is not None:
            self.save_and_reset()
            self.new_file.close()
            self.new_file = None

        print(f"All files combined. Created {self.new_file_idx} new files.")
        
class HDF5CombinerDownstream:
    def __init__(self, src_paths, out_dir, max_file_size=2000):
        self.src_paths = src_paths
        self.out_dir = out_dir
        self.max_file_size = max_file_size  # in MB
        self.new_file_idx = 0
        self.current_file_size = 0  # in MB
        self.new_file = None
        
        self.data_idx = 0
                
        self.shape = self.get_shape()
        self.descriptions = self.get_descriptions()
                
        # Estimate the number of windows based on shape, data type and max file size
        self.n_windows = int(self.max_file_size * 1e6 / (np.prod(self.shape) * np.dtype(np.float32).itemsize))
        
        
    def get_shape(self):
        with h5py.File(self.src_paths[0], 'r') as file:
            return file['data'].shape[1:]
        
    def get_descriptions(self):
        with h5py.File(self.src_paths[0], 'r') as file:
            return file.attrs['descriptions']
        
    def update_files_and_idxs(self, files, idxs):
        
        idxs_to_remove = [i for i in range(len(files)) if i not in idxs]
        # Reverse sort the idxs_to_remove to ensure we remove from end to start,
        # which prevents index shift issues during removal.
        for idx in sorted(idxs_to_remove, reverse=True):
            del files[idx]

        # Create a dictionary to map old indexes to new ones
        # Since files at positions in idxs_to_remove are removed, the indexes after those
        # need to be decremented by the number of removals up to that point.
        index_map = {}
        removals_count = 0
        for i in range(len(files) + len(idxs_to_remove)): # original length of files before removal
            if i in idxs_to_remove:
                removals_count += 1
            else:
                index_map[i] = i - removals_count

        # Update idxs based on the new index mapping
        updated_idxs = [index_map[idx] for idx in idxs if idx in index_map] # Ensure idx is in the new mapping
        
        return np.array(files, dtype=h5py.string_dtype()), np.array(updated_idxs, dtype=np.int32)

    def create_new_file(self):
        if self.new_file is not None:
            self.new_file.close()
            
        #file_path = join(self.out_dir, f'combined_{self.new_file_idx}.hdf5')
        # make file_path be combined_00001.hdf5, combined_00002.hdf5, etc.
        file_path = join(self.out_dir, f'combined_{self.new_file_idx:05d}.hdf5')
        
        if Path(file_path).exists():
            raise ValueError(f"File {file_path} already exists. Please delete it and try again.")
        else:
            self.new_file = h5py.File(file_path, 'w')
        
        # Add checksum Fletcher32 
        self.new_file.create_dataset(
            "data", shape=(self.n_windows, *self.shape), chunks=(1, *self.shape),
            maxshape=(None, *self.shape), dtype=np.float32, fletcher32=True
            )
        
        self.new_file_idx += 1
        
        self.file_idxs = np.empty((0), dtype=np.int32)
        self.files = np.empty((0), dtype=h5py.string_dtype())
        self.time_slices = np.empty((0, 2), dtype=np.float32)
        self.labels = np.empty((0), dtype=np.int32)
        
        self.current_file_size = 0
        self.data_idx = 0

    def save_and_reset(self):        
        
        # Resize the dataset to the correct size
        self.new_file['data'].resize((self.data_idx, *self.shape))
        
        self.files, self.file_idxs = self.update_files_and_idxs(self.files.tolist(), self.file_idxs.tolist())
        self.new_file.create_dataset("labels", data=self.labels, dtype=np.int32, fletcher32=True)
        
        # Save indices as a dataset
        self.new_file.create_dataset("file_idxs", data=self.file_idxs, dtype=np.int32, fletcher32=True)
        # self.new_file.attrs['file_idxs'] = self.file_idxs
        
        # Save time slices as a dataset
        self.new_file.create_dataset("time_slices", data=self.time_slices, dtype=np.float32, fletcher32=True)
        # self.new_file.attrs['time_slices'] = self.time_slices   
    
        self.new_file.attrs['files'] = self.files
        self.new_file.attrs['descriptions'] = self.descriptions
        
        assert len(self.time_slices) == len(self.file_idxs) == self.data_idx 

    def add_data(self, data, file_idxs, files, time_slices, labels):
        data_size = data.nbytes / 1e6  # Convert bytes to MB
        if self.current_file_size + data_size > self.max_file_size:
            self.save_and_reset()
            self.create_new_file()
    
        current_size = self.new_file['data'].shape[0]
        n_elem = data.shape[0]
        
        # Resize the dataset while self.data_idx+n_elem is out of bounds
        while self.data_idx+data.shape[0] > current_size:
            new_size = current_size + self.n_windows
            self.new_file['data'].resize((new_size, *self.shape))
            current_size = self.new_file['data'].shape[0]
            print(f"Resized dataset to {new_size}")
        
        # Add the data
        self.new_file['data'][self.data_idx:self.data_idx+n_elem] = data
        self.data_idx += n_elem
        
        # Fix the indices
        if len(self.file_idxs) > 0:
            file_idxs = file_idxs + len(self.files) 
                
        if len(file_idxs) > 0:
            self.file_idxs = np.concatenate([self.file_idxs, file_idxs], axis=0)
            
        self.files = np.concatenate([self.files, files], axis=0)
        
        if len(time_slices) > 0:
            self.time_slices = np.concatenate([self.time_slices, time_slices], axis=0)
            
        if len(labels) > 0:
            self.labels = np.concatenate([self.labels, labels], axis=0) 
        
        self.current_file_size += data_size

    def combine(self):
        self.create_new_file()  # Prepare the first new file
        
        for path in tqdm(self.src_paths):
            with h5py.File(path, 'r') as file:
                # Extract data and attributes
                data = file['data'][:]
                file_idxs = file.attrs['file_idxs']
                files = file.attrs['files']
                time_slices = file.attrs['time_slices']
                
                labels = file['labels'][:]
                self.add_data(data, file_idxs, files, time_slices, labels)

        # Close the last file properly
        if self.new_file is not None:
            self.save_and_reset()
            self.new_file.close()
            self.new_file = None

        print(f"All files combined. Created {self.new_file_idx} new files.")

        
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Combine HDF5 files')
    parser.add_argument('src_dir', type=str, help='Directory containing HDF5 files')
    parser.add_argument('out_dir', type=str, help='Output directory')
    parser.add_argument('--max_file_size', type=int, default=2000, help='Maximum file size in MB')
    
    args = parser.parse_args()
    data_files_paths = glob(join(args.src_dir, '*.hdf5'))
    data_files_paths.sort()
    
    out_dir = args.out_dir
    src_dir = args.src_dir
    max_file_size = args.max_file_size

    data_files_paths = glob(join(src_dir, '**/*.hdf5'), recursive=True)
    data_files_paths.sort()

    # Create directory if it does not exist
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    #combiner = HDF5CombinerDownstream(data_files_paths, out_dir, max_file_size=max_file_size)
    combiner = HDF5Combiner(data_files_paths, out_dir, max_file_size=max_file_size)
    combiner.combine()
    
    # Delete the original dir
    import shutil
    shutil.rmtree(src_dir)
    
    # Rename the new dir
    import os
    os.rename(out_dir, src_dir)
    
    #combiner = HDF5Combiner(data_files_paths, out_dir, max_file_size=max_file_size)
    #combiner.combine()
