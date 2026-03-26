import numpy as np
from numpy import pi
a = np.arange(10).reshape(2, 5)    
print(a)
print(a.shape)
print(a.ndim)
print(a.size)
print(a.dtype)

b= np.array([1,2])
print(b.shape)
print(b.ndim)
print(b.size)

b = np.array([(1.5, 2, 3), (4, 5, 6)])
print(b)
print(b.shape)  

c = np.array([[1, 2], [3, 4]], dtype=np.int64)
print(c)

d = np.zeros((3, 4))
print(d)
f = np.empty((2, 3)) 
print(f)    

ar = np.arange(10, 30, 5)
print(ar)   

z = np.linspace(0, 2, 9)
print(z)

c = np.arange(24).reshape(2, 3, 4)
print(c)

print("a and b are calculated as follows:")
a = np.array([20, 30, 40, 50])

b = np.arange(4)

print(a)
print(b)    

c = a - b
print(c)
b**2
print(b**2)
print (10 * np.sin(a))

A = np.array([[1, 1],
              [0, 1]])
B = np.array([[2, 0],
              [3, 4]])  
print(A * B)
print(A @ B)


rg = np.random.default_rng(1)  # create instance of default random number generator
a = np.ones((2, 3), dtype=int)
b = rg.random((2, 3))
print(a)
print(b)

a *= 3  
print(a)
b += a  

print(a)

print(b)


#When operating with arrays of different types, the type of the resulting array corresponds to the more general or precise one (a behavior known as upcasting).

a = np.ones(3, dtype=np.int32)
b = np.linspace(0, pi, 3,dtype=np.int16)   
print(a)
print(b)

#Many unary operations, such as computing the sum of all the elements in the array, are implemented as methods of the ndarray class.

a = rg.random((2, 3),dtype=np.float32)

print(a)
print(a.sum())
print(a.min())  
print(a.max())

#By default, these operations apply to the array as though it were a list of numbers, regardless of its shape. However, by specifying the axis parameter you can apply an operation along the specified axis of an array:
b = np.arange(12).reshape(3, 4)
print(b)
print(b.sum(axis=0))
print(b.sum(axis=1))
print(b.cumsum(axis=1))

print("Universal Functions")

B = np.arange(4)
print(B)
print(np.exp(B))
print(np.sqrt(B))   

print("Indexing, Slicing and Iterating")

a = np.arange(10)**3
print(a)
print(a[2]) 
print(a[2:5])
print(a[2:3:1])

print(a[::-1])

for i in a:
    print(i**(1 / 3.))

print("Multidimensional arrays can have one index per axis. These indices are given in a tuple separated by commas:")
def f(x, y):
    return 10 * x + y
b = np.fromfunction(f, (5, 4), dtype=int)
print(b)
print(b[2, 3])
print(b[0:4, 2])
print(b[:2, 1])

print("special slicing: ::-1 means to take all elements in the array in reverse order.")

print(b[1:4, ])

a = np.arange(10)**3
print(a)
a = a[-2:7:-1]
print(a)

print("... test ")

c = np.array([[[  0,  1,  2],  # a 3D array (two stacked 2D arrays)
               [ 10, 12, 13]],
              [[100, 101, 102],
               [110, 112, 113]]])

print(c)
print(c.shape)
print(c[:, 0:2, 0:3])

print("Iterating")

for element in b.flat:
    print(element)

print("Shape manipulation") 

a = np.floor(10 * rg.random((3, 4)))
print(a)
print(a.T)
print(a.reshape(2, 6) )
print(a.ravel())
print(a.reshape(6, 2) )


print("Stacking together different arrays")

a = np.floor(10 * rg.random((2, 2)))
print(a)

b = np.floor(10 * rg.random((2, 2)))
print(b)

print(np.vstack((a, b)))
print(np.hstack((a, b)))

print("Column stack")
from numpy import newaxis
print(np.column_stack((a, b)))  # with 2D arrays

a = np.array([4., 2.])
b = np.array([3., 8.])
print(np.column_stack((a, b))) 

print(a)
print(a[:, newaxis])

print(np.hstack((a[:, newaxis], b[:, newaxis])) )

print("Splitting one array into several smaller ones")
a = np.floor(10 * rg.random((2, 12)))
print(a)
print(np.hsplit(a, 3))  # Split a into 3 equal arrays along the 2nd axis (zero-based)

print("No Copy at all - views")

a = np.array([[ 0,  1,  2,  3],
              [ 4,  5,  6,  7],
              [ 8,  9, 10, 11]])

b = a  
print(b is a ) 

print(f"value of a is : \n {a} and value of b is: \n {b}")

c=a.view()
print(c is a)
c = c.reshape(2, 6)
print(c)

c[0,5] = 1234
print(c)
print(a)

a = np.arange(int(1e8))
print(a)
b = a[:100].copy()

del a
print(b)

print("Indexing with arrays of indices")
a = np.arange(12)**2  # the first 12 square numbers
i = np.array([1, 1, 3, 8, 5])  # an array of indices

print(a)    
print(a[i])  # the elements of `a` at the positions `i`

print("pallet testing")
palette = np.array([[0, 0, 0],         # black
                    [255, 0, 0],       # red
                    [0, 255, 0],       # green
                    [0, 0, 255],       # blue
                    [255, 255, 255]]) 

image = np.array([[0, 1, 2, 0],  # each value corresponds to a color in the palette
                  [0, 3, 4, 0]])

print(palette[image]) 

print("reshapeing an array")
a = np.arange(12).reshape(3, 4)
print(a)

b=a>4
print(b)    
print(a[b])


print("IMAGE PRINTING TESTING   ")

import numpy as np
import matplotlib.pyplot as plt
def mandelbrot(h, w, maxit=20, r=2):
    """Returns an image of the Mandelbrot fractal of size (h,w)."""
    x = np.linspace(-2.5, 1.5, 4*h+1)
    y = np.linspace(-1.5, 1.5, 3*w+1)
    A, B = np.meshgrid(x, y)
    C = A + B*1j
    z = np.zeros_like(C)
    divtime = maxit + np.zeros(z.shape, dtype=int)

    for i in range(maxit):
        z = z**2 + C
        diverge = abs(z) > r                    # who is diverging
        div_now = diverge & (divtime == maxit)  # who is diverging now
        divtime[div_now] = i                    # note when
        z[diverge] = r                          # avoid diverging too much

    return divtime
plt.clf()
plt.imshow(mandelbrot(400, 400))