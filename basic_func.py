import quimb.tensor as qtn
import numpy as np

# Basic Data Structure
arrays = []
to_contract = [2,3,5]
for i in range(10):
    if i == 0 or i == 9:
        data = np.random.rand(2, 2)
        inds = (f'k{i}', f'k{i+1}')
    else:
        data = np.random.rand(2, 2, 2)
        inds = (f'k{i}', f'k{i+1}', f'j{i}')
    tags = [f'node_{i}']
    if i in to_contract:
        tags.append('to_contract')
    arrays.append(qtn.Tensor(data=data, inds=inds, tags=tags)) #arr -> tensor

tensor_train = arrays[0]
for i in range(1, 10):
    tensor_train = tensor_train & arrays[i] #tensor -> tensor_network (tensor_train here)

print(tensor_train)
tensor_train.draw(figsize=(4, 4), show_inds='bond_size')

# Contraction
contracted_TT = tensor_train.contract(all, optimize='auto-hq')
contracted_TT.draw(show_inds='all')

selected_contraction_TT = tensor_train^'to_contract'
selected_contraction_TT.draw(show_inds='all')

# Compression
ta = qtn.rand_tensor([4, 4, 10], ['a', 'b', 'c'], 'A')
tb = qtn.rand_tensor([10, 4, 4, 4], ['c', 'd', 'e', 'f'], 'B')
print("tensor_a: ", ta)
print("tensor_b: ", tb)
(ta & tb).draw(['A', 'B'], figsize=(4, 4), show_inds='bond-size')
qtn.tensor_compress_bond(ta, tb, max_bond=2, absorb='left')
(ta & tb).draw(['A', 'B'], figsize=(4, 4), show_inds='bond-size')