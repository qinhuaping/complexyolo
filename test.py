import numpy as np
import cv2
import math

a = np.array([[1, 2, 3, 4],
			  [2, 3, 4, 1],
			  [3, 3, 4, 5],
			  [5, 5, 6, 6]])
#indices = np.lexsort((a[:,0],a[:,1],a[:,2]))
indices = [0,2,1,3]
print((a[:,0],a[:,1],a[:,2]))
print("indices::",indices)
a=a[[0,2,1,3]]
print("a",a)

b = np.array([[1,1,2],[2,3,3],[1,3,4],[3,1,2]])
s,indice = np.unique(b,axis = 0, return_index=True)

print indice
print b[indice]
