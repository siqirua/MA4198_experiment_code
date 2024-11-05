import numpy as np
import quimb.tensor as qtn
import time
import csv

def create_sin_tensor(n, precision):
    # Step 1: Create the grid points (0, 0.1, 0.2, ..., 1.0)
    grid = np.arange(0, 1 + precision, precision)
    # Generate random coefficients a1, a2, ..., an within range [amin, amax]
    a = amin + (amax - amin) * np.random.rand(n)
    # Step 3: Initialize the tensor for f(x1, ..., xn)
    tensor_size = [len(grid)] * n  # Tensor will be grid^n in size
    tensor_values = np.zeros(tensor_size)
    
    # Step 4: Populate the tensor by evaluating sin(a1*x1 + ... + an*xn) for each grid point combination
    for idx in np.ndindex(*tensor_size):  # Iterate over all possible combinations of grid points
        x = np.array([grid[i] for i in idx])  # Get the grid values x1, x2, ..., xn
        tensor_values[idx] = np.sin(np.dot(a, x))  # Compute sin(a1*x1 + a2*x2 + ... + an*xn)

    # Step 5: Create the Quimb tensor with the appropriate indices
    tensor_inds = [f'x{i+1}' for i in range(n)]  # Create indices x1, x2, ..., xn
    tensor = qtn.Tensor(tensor_values, inds=tensor_inds, tags=['f1'])
    #tensor = qtn.TensorNetwork([tensor])  # Convert the tensor to a tensor network
    return tensor, a, grid

# Example usage

precision = 0.1  # Precision for grid points
amin = 0  # Minimum coefficient value
amax = 1  # Maximum coefficient value


def create_and_decompose_sin_func_tensor(n):
    tensor, a, grid = create_sin_tensor(n, precision)

    tensors = []
    remaining_tensor = tensor


    for i in range(n - 1):
        if i == 0:
            split_ind = 1
        else:
            split_ind = 2
        left_inds = list(remaining_tensor.inds[:split_ind]) # Split on the current index
        right_inds = list(remaining_tensor.inds[split_ind:]) # Keep remaining indices for the right tensor
        # Perform the split
        left_tensor, remaining_tensor = remaining_tensor.split(left_inds, right_inds = right_inds, cutoff=1e-6, bond_ind = f'b{i+1}')

        # Add the left tensor to the list of decomposed tensors
        tensors.append(left_tensor)

    # Append the final remaining tensor
    tensors.append(remaining_tensor)
    tensor_train = qtn.TensorNetwork(tensors)
    tensor_list = tensor_train.tensors
    ranks = [tensor.shape[0] for tensor in tensor_list[:-1]] + [tensor_list[-1].shape[1]]  # Last tensor shape[1]

    return tensor_train

def tt_sum(tt1, tt2):
    time_start = time.time()
    sum_tt = qtn.tensor_network_sum(tt1, tt2)
    time_end = time.time()
    #print(time_start, time_end)
    return sum_tt, time_end - time_start

# Prepare to save results
csv_file = 'tensor_times_sum_repeat.csv'
unit_tensor =  create_and_decompose_sin_func_tensor(6)
start_tensor = unit_tensor.copy()

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['n', 'summation_time'])  # Header row
    n = 1
    while True:
        try:
                #add_tensor = create_and_decompose_sin_func_tensor(6)
                start_tensor, time_taken = tt_sum(start_tensor, unit_tensor)
                writer.writerow([n, time_taken])
                print(f"n={n}, summation_time={time_taken}")
        except:
            break
        finally:
            n += 1

print(f"Results saved to {csv_file}")