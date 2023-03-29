import os
import numpy as np

print("started")
general_path = "/home/kryc1/stories2image/data/subspaces/general/"
computer_path = "/home/kryc1/stories2image/data/subspaces/computer_internet/"
education_path = "/home/kryc1/stories2image/data/subspaces/education_reference/"
science_path = "/home/kryc1/stories2image/data/subspaces/science_math/"

# Get a list of file paths in general_path
file_paths = [os.path.join(education_path, f) for f in os.listdir(education_path) if os.path.isfile(os.path.join(education_path, f))]

# Sort the list of file paths by name
file_paths.sort()

# Create an empty list to store the NumPy arrays
arrays = []

# Load each file into a NumPy array and append to the list
cnt = 0
for file_path in file_paths:
    array = np.loadtxt(file_path)
    arrays.append(array)
    cnt = cnt + 1
    if cnt % 100 == 0:
    	print(cnt)
print(cnt)

# Convert the list of NumPy arrays into a NumPy array of shape (num_files, num_rows, num_cols)
arrays = np.stack(arrays)

# Save the NumPy array to disk in a binary format
np.save("/home/kryc1/stories2image/data/subspaces_npy/education_reference.npy", arrays)
print("ended")
