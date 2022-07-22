import matplotlib.path as mplPath
import numpy as np
import matplotlib.pyplot as plt
#poly = [190, 50, 500, 310]
poly_path = mplPath.Path(np.array([[0, 10*np.sqrt(3)/2],
                                    [10/2, 0],
                                    [10*3/2, 0]]))
x = np.random.uniform(low=0, high = 3*10/2, size = (1000,))
y = np.random.uniform(low=0, high = np.sqrt(3)*10/2, size = (1000,))
xy = np.column_stack((x,y))
print (xy.shape)
idx = poly_path.contains_points(xy)
plt.scatter(x[idx], y[idx], s = 2)
plt.show()
#print(point, " is in polygon: ", poly_path.contains_point(point))

#point = (1200, 1000)
#print(point, " is in polygon: ", poly_path.contains_point(point))