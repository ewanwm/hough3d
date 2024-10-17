## hough3d

![Animated gif showing the results on a test dataset.](https://raw.githubusercontent.com/Jfeatherstone/hough3d/refs/heads/master/test_images/synthetic_b.gif)

This is a python library to perform a Hough transform to detect lines in 3D point clouds. We implement the iterative algorithm described in Dalitz et al. (2017):

1. Direction vectors in 3D are discretized as the position vectors for the vertices of a tessellated icosahedron.
2. All points in the point cloud vote, generating the Hough space representation.
3. The highest voted line is selected, and the points surrounding that line are identified.
4. Linear regression is performed on the selected points to refine the line parameters.
5. All of the votes contributed by the selected points are subtracted from Hough space.
6. Steps 3-5 are repeated until there are no satisfactory lines remaining.

This method has the advantage of being able to identify numerous lines without the computational burden of recalculating the entire Hough space each time.

The authors of the cited paper have implemented a C++ version [here](https://github.com/cdalitz/hough-3d-lines/). I am currently working on trying to improve the algorithm specifically to detect line segments accurately, but for now this library functions equivalently to the C++ one. Functions are transpiled with [numba](https://numba.readthedocs.io/), so there shouldn't be a huge difference in computation time despite this one being written in Python.


### Usage

For test data, I would recommend looking at the [datasets in the C++ repo](https://github.com/cdalitz/hough-3d-lines/tree/master/data).

```
from hough3d import hough3D

data = np.genfromtxt('data.csv', delimiter=',')

# Determines how many tesselations of the icosahedron
# to perform; larger number means more fine direction
# discretization, but larger computation time.
# See genIcosahedron docs for exact numbers.
directionGranularity = 4
directionVectors = genIcosahedron(directionGranularity)

# See docs for specific info on kwargs; generally
# these need to be chosen for each specific dataset.
linePoints = hough3D(data, directionVectors, latticeSize=128,
                     neighborDistance=0.01, minPointsPerLine=5)

```

And then visualize the results:

```
import open3d as o3d

flattenedLinePoints = linePoints.reshape((len(linePoints)*2, 3))

lineSet = o3d.geometry.LineSet()
lineSet.points = o3d.utility.Vector3dVector(flattenedLinePoints)
lineSet.lines = o3d.utility.Vector2iVector([[i,i+1] for i in range(len(flattenedLinePoints))[::2]])

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(data)

o3d.visualization.draw_geometries([lineSet, pcd])
```
