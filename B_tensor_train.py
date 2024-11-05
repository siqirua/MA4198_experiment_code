import numpy as np
from read_matrix import read_matrix
import itertools
import quimb.tensor as qtn
import time

B = read_matrix('B.txt')

def generate_all_pairings(arr):
    # Base case: if the list is empty, return an empty pairing
    if len(arr) == 0:
        return [[]]
    
    # Take the first element and pair it with every other element
    first = arr[0]
    pairings = []
    
    # For each pair with the first element, recursively pair the rest
    for i in range(1, len(arr)):
        pair = [first, arr[i]]
        rest = arr[1:i] + arr[i+1:]  # Elements left after removing the paired elements
        # Recursively generate pairings for the rest
        for rest_pairing in generate_all_pairings(rest):
            pairings.append([pair] + rest_pairing)
    
    return pairings

# Example input
arr = ['i1', 'i2', 'i3', 'i4', 'i5', 'i6']

# Generate all pairings
pairings = generate_all_pairings(arr)
print(pairings)
# Print the results
for pairing in pairings:
    print(pairing)

def product(pairing, B):
    # Initialize the product as 1
    prod = 1
    # Multiply the elements in each pair
    for pair in pairing:
        prod *= B[pair[0], pair[1]]
    return prod

def generate_tensor_L2(B):
    dim = B.shape[0]
    L = np.zeros((dim, dim), dtype=np.complex128)
    
    for i1 in range(dim):
        for i2 in range(dim):
            L[i1, i2] = sum(product(pairing, B) for pairing in generate_all_pairings([i1, i2]))
    return L


def generate_tensor_L4(B):
    dim = B.shape[0]
    L = np.zeros((dim, dim, dim, dim), dtype=np.complex128)
    
    for i1 in range(dim):
        for i2 in range(dim):
            for i3 in range(dim):
                for i4 in range(dim):
                    L[i1, i2, i3, i4] = sum(product(pairing, B) for pairing in generate_all_pairings([i1, i2, i3, i4]))
    return L

def generate_tensor_L4_alternative(B):
    N = B.shape[0]
    indices = ['i1', 'i2', 'i3', 'i4']

    pairings = [
        [('i1', 'i2'), ('i3', 'i4')],
        [('i1', 'i3'), ('i2', 'i4')],
        [('i1', 'i4'), ('i2', 'i3')],
    ]

    tensor_networks = []

    for pairing in pairings:
        tensors = []
        involved_indices = set()

        # Create B tensors for each pair
        for (ind_a, ind_b) in pairing:
            tB = qtn.Tensor(B, inds=[ind_a, ind_b])
            tensors.append(tB)
            involved_indices.update([ind_a, ind_b])

        # Add identity tensors for any indices not involved in B
        all_indices = set(indices)
        uninvolved_indices = all_indices - involved_indices

        for ind in uninvolved_indices:
            tI = qtn.Tensor(np.eye(N), inds=[ind, ind])
            tensors.append(tI)

        # Form the tensor network
        tn = qtn.TensorNetwork(tensors)
        tensor_networks.append(tn)

    # Sum all tensor networks
    L_tn = tensor_networks[0].copy()

    for tn in tensor_networks[1:]:
        print(L_tn)
        print(tn)
        L_tn = qtn.tensor_network_sum(L_tn, tn)  # Use tensor_network_sum

    # Contract the tensor network into a single tensor
    L_tensor = L_tn.contract(all, output_inds=indices)

    return L_tensor


def generate_tensor_L6(B):
    dim = B.shape[0]
    L = np.zeros((dim, dim, dim, dim, dim, dim), dtype=np.complex64)
    
    for i1 in range(dim):
        for i2 in range(dim):
            for i3 in range(dim):
                for i4 in range(dim):
                    for i5 in range(dim):
                        for i6 in range(dim):
                            L[i1, i2, i3, i4, i5, i6] = sum(product(pairing, B) for pairing in generate_all_pairings([i1, i2, i3, i4, i5, i6]))
    return L

def generate_tensor_L6_alternative(B):
    indices = ['i1', 'i2', 'i3', 'i4', 'i5', 'i6']
    pairings = generate_all_pairings(indices)
    def create_tensor_network_for_pairing(pairing, N, B):
        """
        Create a tensor network for a given pairing with standardized indices.
        """
        # List to hold all tensors in the term
        tensors = []

        # Standard index order
        standard_indices = ['i1', 'i2', 'i3', 'i4', 'i5', 'i6']

        # Map from actual indices to standard indices
        index_map = {}
        involved_indices = set()

        # Assign standard indices to the indices in the pairing
        idx_counter = 0
        for inds in pairing:
            for ind in inds:
                if ind not in index_map:
                    index_map[ind] = standard_indices[idx_counter]
                    idx_counter += 1
                involved_indices.add(ind)

        # Map for uninvolved indices
        all_indices = set(standard_indices)
        uninvolved_indices = all_indices - set(index_map.values())

        # Add B tensors for each pair in the pairing
        for inds in pairing:
            # Reindex the indices to standard indices
            new_inds = (index_map[inds[0]], index_map[inds[1]])
            tB = qtn.Tensor(B, inds=new_inds)
            tensors.append(tB)

        # Add identity tensors for uninvolved indices
        for ind in uninvolved_indices:
            data = np.ones(N)
            tI = qtn.Tensor(data, inds=(ind,))
            tensors.append(tI)

        # Create the tensor network
        tn = qtn.TensorNetwork(tensors)
        return tn

    # List to hold tensor networks for all terms
    tn_terms = []

    # Build tensor networks for each term
    for pairing in pairings:
        tn = create_tensor_network_for_pairing(pairing, 51, B)
        tn_terms.append(tn)

    # Sum the tensor networks
    # Initialize L_tn with the first term
    L_tn = tn_terms[0].copy()

    # Add the rest of the terms
    for tn in tn_terms[1:]:
        print(L_tn)
        print(tn)
        L_tn.draw(show_inds='all')
        tn.draw(show_inds='all')
        L_tn = qtn.tensor_network_sum(L_tn, tn) # TensorNetwork addition

    return L_tn

def tensor_train_decompose(tensor, n):
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
        left_tensor, remaining_tensor = remaining_tensor.split(left_inds, right_inds = right_inds, cutoff=1e-6)

        # Add the left tensor to the list of decomposed tensors
        tensors.append(left_tensor)

    # Append the final remaining tensor
    tensors.append(remaining_tensor)
    tensor_train = qtn.TensorNetwork(tensors)
    tensor_list = tensor_train.tensors
    ranks = [tensor.shape[0] for tensor in tensor_list[:-1]] + [tensor_list[-1].shape[1]]  # Last tensor shape[1]

    return tensor_train, ranks

def check_decomposition(TT, original_tensor):
    # Compute the approximation error
    approx_tensor = TT.contract()
    # Convert the Quimb tensor objects to dense NumPy arrays before computing the norm
    original_tensor_dense = original_tensor.data
    approx_tensor_dense = approx_tensor.data

    # Now you can calculate the error using NumPy's norm
    error = np.linalg.norm(original_tensor_dense - approx_tensor_dense) / np.linalg.norm(original_tensor_dense)
    
    # Get the ranks of the tensor train decomposition
    ranks = [tensor.shape[0] for tensor in TT.tensors[:-1]] + [TT.tensors[-1].shape[1]]
    
    return error, ranks

start_time = time.time()
tensor_values = generate_tensor_L4(B)
tensor_inds = [f'x{i+1}' for i in range(4)] 
tensor = qtn.Tensor(tensor_values, inds=tensor_inds, tags=['f1'])
tensor_alt = generate_tensor_L4_alternative(B)
print(tensor_alt)
tensor_alt = tensor_alt.contract(all, output_inds=['i1', 'i2', 'i3', 'i4'])
print(tensor)
print(tensor_alt)
print(tensor.data[0][0])
print(tensor_alt.data[0][0])
#print(tensor.data - tensor_alt.data)

'''
generation_time = time.time() - start_time

convert_time = time.time() - start_time - generation_time
tensor_train, ranks = tensor_train_decompose(tensor, 2)
decompose_time = time.time() - start_time - generation_time - convert_time
print(tensor_train)
print(ranks)
tensor_train.draw(figsize=(4, 4), show_inds='bond_size')
error, ranks = check_decomposition(tensor_train, tensor)
print(f"Approximation error: {error}")
print(f"Ranks: {ranks}")
print(f"Generation time: {generation_time}")
print(f"Convert time: {convert_time}")
print(f"Decompose time: {decompose_time}")
'''
