import torch 

device = "cuda" if torch.cuda.is_available() else "cpu"

my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32
                , device=device, requires_grad=True)
print(my_tensor)

# print dtype = torch.float32
print(my_tensor.dtype)

print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.shape)
print(my_tensor.requires_grad)

'''
Different ways of initialising tensors
'''

# initialises the tensor with uninitialised values
# creates a tensor but does not initialise the values
# memory is only allocated once the values are given
x = torch.empty(size=(3, 3))
print(x)

# initialises the tensor will all zero values
x = torch.zeros((3, 3))

# initialise value from a unifrom distribution
# in between values 0 and 1
x = torch.rand((3, 3))

# 3x3 matrix of all values 1
x = torch.ones((3, 3))

# create an identiy matrix on the tensor
x = torch.eye(5, 5)

# same as arange function in python
x = torch.arange(start = 0, end= 5, step =1)
print(x)

# start at 0.1 and end at 1 with 10 values in between
x = torch.linspace(start=0.1, end=1, steps=10)
print(x)

# get values from the normal distribution
x = torch.empty(size=(1, 5)).normal_(mean=0, std=1)

# get values from the uniform distribution
# lower = 0 and upper = 1 here
x = torch.empty(size=(1, 5)).uniform_(0, 1)

# 3x3 matrix with diagonal all ones
# similar to creating an identity matrix
x = torch.diag(torch.ones(3))

'''
Different ways to convert tensros to other types (int, float, double)
'''
tensor = torch.arange(4) # Creates [0, 1, 2, 3]
print(tensor.bool()) # converts to bool ie. [False, True, True, True]

print(tensor.short()) # converts it to int16
print(tensor.long()) # converts it to int64 (Important)

print(tensor.half()) # convert it to float16
print(tensor.float()) # float32 (Imporatant) used it most conventions
print(tensor.double()) # float64

'''
Array to Tensor conversion and vice versa
'''
import numpy as np 

np_array = np.zeros((5, 5))
# convert to torch tensor
tensor = torch.from_numpy(np_array)

# convert back to numpy array
np_array_back = tensor.numpy()

'''
Tensor Math
'''

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

# Addition:
z = torch.empty(3)
torch.add(x, y, out=z)

z = torch.add(x, y)
z = x + y

# Subtraction
z = x-y

# Division
z = torch.true_divide(x, y)

# inplace operations
t = torch.zeros(3)

# methods with an underscore in them
# are in place hence more memory efficient
t.add_(x)

t += x # also in place, behaves similar to t.add_(x)

# Exponentation
z = x.pow(2) # power of 2 to every element of x
print(z)

z = x**2 # two asterics does the same

# Simple Comparisions
z = x>0
print(z) # Prints all true as all elements are > 0

# Matrix Multiplication
x1 = torch.rand((2, 3))
x2 = torch.rand((3, 5))

# both of these are ways to matrix multiplication
x3 = torch.mm(x1, x2)
x3 = x1.mm(x2)

# Matrix Exponentation
matrix_exp = torch.rand(5, 5)
matrix_exp.matrix_power(3) # equivalent to matrix applied to itself 3 times

# Element by multiplication
z = x*y
print(z) # Element wise multiplication

# dot product is basically element wise multiplication
# then we take the sum of it all
z = torch.dot(x, y)

# Batch Matrix Multiplication
batch = 32
n= 10
m = 20
p = 30

# when our tensor has more than 2 dimension then we have to incorporate batch
# for 2d matricies we can do just normal multiplication
# but for higher dimensional matricies we use batch matrix multiplication
tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p)) # the multiplication is going to be done 
# along the common dimention ie. n
out_bmm = torch.bmm(tensor1, tensor2)
# results tensor will have dimensions: (batch, n, p)

'''
Broadcasting : ability to perform element wise operations
of tensors/arrays of different sizes and shapes
'''
x1 = torch.rand(5, 5)
x2 = torch.rand(1, 5)

# x2 is going to be broadcastes 5 times like 5 rows
# then then element wise subtraction takes place
z = x1 - x2

# specify which dimension to sum over
sum_x = torch.sum(x, dim=0)

# returns the maximum values and their indices
# we can specify which dimension we take the max over
values, indices = torch.max(x, dim=0)
values, indices = torch.min(x, dim=0)

# going to take the absolute value for each
# element in the tensor
abs_x = torch.abs(x)

# returns the index of the class that is maximum
z = torch.argmax(x, dim=0)
z = torch.argmin(x, dim=0)

# we can also find the mean of the tensor along a dimension
# but mean requires us for the tensor to be of float32() values
mean_x = torch.mean(x.float(), dim=0)

# element wise comparision
z = torch.eq(x, y) # returns [False, False, False]

'''
Sorting
'''
# Along an axis, sorts the elements 
sorted_y, indices = torch.sort(y, dim=0, descending=False)

# returns the sorted_y and also the indices that we need to
# swap for the matrix to get sorted

# check all indices of x and clamp them
# any value greater than 10, clamp it to 10
# and any value smaller than 0 becomes 0
z = torch.clamp(x, min=0, max=10)

x = torch.tensor([1, 0, 1, 1, 1], dtype=torch.bool)
z = torch.any(x) # this checks if any values is true
print(z)

z = torch.all(x) # this check is all values are true

'''
Tensor Indexing
'''
print("-------- Tensor Indexing -----------")
batch_size = 10
features = 25
x = torch.rand((batch_size, features))

print(x[0, :].shape) 
# x[0:,:] means we want the first row of our batch 
# and within it we want all the features along that row

print(x[:, 0].shape)
# now we will get the first feature of all the examples
# basically we will get the entire column

# get the 3 example of our batch get the first 10 features
print(x[2, :10])

# Assign our tensors
x[0, 0] = 100

'''
Fancy Tensor Indexing
'''
x = torch.arange(10)
indices = [2, 5, 8] # Pick out values exactly matching the indices
print(x[indices])

# pick out elements
x = torch.rand((3, 5))
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(x[rows, cols])

# advanced indexing
x = torch.arange(10)
print(x[(x < 2) | (x > 8)]) # pick out all the elements less than 2 or greater than 8

print(x[x.remainder(2)==0]) # pick out all the even elements

'''
Other Useful Operations
'''

print(torch.where(x > 5, x, x*2))
# what this does is the first arguemtn is the condition
# if the condition is true we return the second argument
# if the condition is false we return the third argument

print(torch.tensor([0, 0, 1, 2, 2, 3, 4]).unique()) # get all unique values of the vector

print(x.ndimension()) # returns the number of dimensions

print(x.numel()) # counts the number of elements in the tensor

'''
Reshape the tensors
'''
print("----- Reshaping Tensors-------")

x = torch.arange(9)
# x_3x3 = x.view(3, 3)
# view and reshape are very similar in their functioning
x_3x3 = x.reshape(3, 3)

print("x_3x3:")
print(x_3x3)

# view acts on contigious tensors stored contigously in memory
# reshape cant act on anyway its always the safer bet but there 
# will be performancy loss 

y = x_3x3.t() # take the transpose of a tensor
# this transpose version is no longer a contigous block of memory
print(y)
print(y.reshape(9))

# do contigous before view or just do reshape directly
# print(y.contigous().view(9))

x1 = torch.rand((2, 9))
x2 = torch.rand((2, 9))
print(torch.cat((x1, x2), dim=0)) # Concatenate the tensors

'''
Unrolling tensors
'''
print("-----Unrolling Tensors----")
# flatten the tensors to a single row
z = x1.view(-1)
print(z)

batch = 64
x = torch.rand((batch, 2, 5))
# preserve the elements along batch
# but flatten the rest 
z = x.view(batch, -1)
print(z.shape) # 64 is preserved and we get 2 x 5 = 10 elements flattened


# permute is like taking the transpose
# if we have two dimensions we take .t()
# but when we have more dimensions we use permute()
z = x.permute(0, 2, 1)

'''
Unsqueeze
'''

x = torch.arange(10)
print(x.unsqueeze(0).shape)
print(x.unsqueeze(1).shape)

# adds a new dimension along 0 and 1
x = torch.arange(10).unsqueeze(0).unsqueeze(1) # 1x1x10

# removes dimensions
z = x.squeeze(1)
print(z.shape)