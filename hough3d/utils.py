import numpy as np

from itertools import product

from scipy.spatial import KDTree

import numba

def genIcosahedron(nSubdivisions=4):
    """
    Generate direction vectors to the vertices of
    a icosahedron subdivided a given amount of times,
    to create a discrete set direction vectors equally
    sampling space.

    Due to not being able to figure out the analytic
    expression for the distance between connected components
    on the shape, only works up to 5 subdivisions. This is
    a hard problem because we can't quite just take the nearest
    neighbor distance because a icosahedron has some vertices
    with 5 neighbors (spaced a distance `a` away) and some
    vertices with 6 neighbors (spaced a distance `b` away, `a`!=`b`).

    Subdivisions | Directions
        0              6
        1              21
        2              81
        3              321
        4              1281
        5              5121

    Parameters
    ----------
    nSubdivisions : int
        The number of times to subdivide the shape, creating a finer
        grid of directions. See table above for number of directions.

    Returns
    -------
    directionVectors : numpy.ndarray[N,3]

    """
    # First, generate the basic icosahedron
    vertices = []
    r = 1/2 * (1 + np.sqrt(5))

    # See equation 4 of Dalitz et al. (2017)
    # This generates all combinations of:
    # (+-1, +-r, 0)
    # (0, +-1, +-r)
    # (+-r, 0, +-1)
    vertexPossibilities = np.array([[1, -1], [0], [r, -r]], dtype=list)

    for i in range(len(vertexPossibilities)):
        rolled = np.roll(vertexPossibilities, i)
        xPossibilities = rolled[0]
        yPossibilities = rolled[1]
        zPossibilities = rolled[2]

        vertices += list(product(xPossibilities, yPossibilities, zPossibilities))

    # Now we have an array of 12 vertices for our basic shape
    vertices = np.array(vertices)

    # Next, we add a vertex between adjacent vertices the
    # requested number of times

    # These were calculated manually, by finding the distance required to make
    # sure every vertex has 5 or 6 neighbors... I'm sure there must be an
    # analytic expression here, but whatever, we probably would only ever
    # use up to 4 or 5 (at the absolute max) so it is what it is...
    # but still TODO
    neighborDistances = [1.0515, 0.6180, 0.3249, 0.1647, 0.0826]
    if nSubdivisions > len(neighborDistances):
        raise Exception(f'Too many subdivisions requested! Only supported up to {len(neighborDistances)}...')

    for i in range(nSubdivisions):

        # Normalize the vertices so they sit on the surface of a sphere
        norms = np.sqrt(np.sum(vertices**2, axis=-1))
        vertices = np.array([vertices[j]/norms[j] for j in range(len(vertices))])
        # Remove duplicates
        vertices = np.unique(vertices, axis=0)

        # We will determine which points are adjacent using a
        # KD Tree; points that are adjecent should be a distance
        # of 2/n away from each other, where n is the current iteration
        # beginning with n=1.
        # Every vertex should have either 5 or 6 neighbors

        # This was used for manually finding the neighbor distances
        #uniq = np.array(np.unique(np.array([np.sqrt(np.sum((vertices - vertices[j])**2, axis=-1)) for j in range(len(vertices))]).flatten(), return_counts=True)).T
        #for u,c in uniq:
        #    print(u, int(c))

        kdTree = KDTree(vertices)
        # Add some small epsilon just in case
        neighbors = kdTree.query_ball_tree(kdTree, neighborDistances[i]*1.15+1e-6)

        # The first point will always be the same point itself, so
        # we need to throw that away
        neighbors = [n[1:] for n in neighbors]
        #print([len(n) for n in neighbors])

        # Now add points between each point and its neighbors
        newVertices = np.array([(np.array(vertices[j]) + np.array(vertices[k]))/2 for j in range(len(neighbors)) for k in neighbors[j]])

        # Add the new vertices to the running array
        vertices = np.concatenate([vertices, newVertices])

    # Final normalization of the vertices
    norms = np.sqrt(np.sum(vertices**2, axis=-1))
    vertices = np.array([vertices[j]/norms[j] for j in range(len(vertices))])

    vertices = np.unique(vertices, axis=0)

    # Now we have to remove mirror images of directions
    # eg. [0, 0, 1] and [0, 0, -1] represent antiparallel
    # slopes, but for our fitting, this difference doesn't matter
    # We do this by marking points as duplicate by reflecting
    # points across each axis, one at a time
    keepVertex = np.ones(len(vertices), dtype=bool)
    for i in range(len(vertices)):
        # We have to first set this to false such that we don't
        # compute the dot product of the vector with itself
        keepVertex[i] = False
        # Now we compute the dot products and make sure none of them
        # are -1 (ie. there exists an antiparallel vector in the list already)
        # Need a small epsilon to account for floating point errors
        keepVertex[i] = np.min(np.dot(vertices[i], vertices[keepVertex].T)) > -1 + 1e-8

    vertices = vertices[keepVertex]

    return vertices


@numba.njit()
def reducedRepConversionMatrices(b):
    """
    Calculate the matrix to convert a line representation
    to it's reduced represetation [1] as:
        ```
        X, Y = reducedRepConversionMatrices(b)
        x' = np.dot(X, p)
        y' = np.dot(Y, p)
        ```

    Parameters
    ----------
    b : numpy.ndarray[3]
        The vector pointing in the direction of the line for which
        we want to calculate the conversion matrices

    Returns
    -------
    X : numpy.ndarray[3]

    Y : numpy.ndarray[3]

    References
    ----------
    [1] K.S. Roberts, A new representation for a line, in Computer Vision
    and Pattern Recognition CVPR’88, 1988, pp. 635–640.

    """
    reducedRepMultX = np.array([1 - b[0]**2/(1 + b[2]),
                               -b[0]*b[1]/(1 + b[2]),
                               -b[0]])

    reducedRepMultY = np.array([-b[0]*b[1]/(1 + b[2]),
                                1 - b[1]**2/(1 + b[2]),
                               -b[1]])

    return reducedRepMultX, reducedRepMultY


@numba.njit()
def rotationMatrix(theta, phi, psi):
    """
    Generate the rotation matrix corresponding to rotating
    a point in 3D space.
    """
    return np.array([[np.cos(theta)*np.cos(psi), np.cos(phi)*np.sin(psi) + np.sin(phi)*np.sin(theta)*np.cos(psi), np.sin(phi)*np.sin(psi) - np.cos(psi)*np.cos(phi)*np.sin(theta)],
                     [-np.cos(theta)*np.sin(psi), np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(theta)*np.sin(psi), np.sin(phi)*np.cos(psi) - np.cos(psi)*np.sin(phi)*np.sin(theta)],
                     [np.sin(theta), -np.sin(phi)*np.cos(theta), np.cos(phi)*np.cos(theta)]])


@numba.njit()
def lineIntersection(a1, b1, a2, b2):
    """
    Compute the intersection of two lines:
        a1 + b1*t and a2 + b2*t
    """
    # Compute the cross product
    crossProd = np.cross(b1, b2)
    if crossProd == 0:
        return np.nan
    # Dot the cross product unit vector into the vector from intercept to intercept
    distance = np.dot(a2 - a1, crossProd) / np.sqrt(np.sum(crossProd**2))
    return distance


@numba.njit()
def distancePointToLine(p, a, b):
    """
    Compute intersection of a point (p) with a line (a + bt).

    Parameters
    ----------
    p : numpy.ndarray[3]
        Point to compute distance to the line

    a : numpy.ndarray[3]
        Point located on the line.

    b : numpy.ndarray[3]
        Direction vector of the line.

    Returns
    -------
    d : float
        Distance from the point to the closest point
        on the line.
    """
    t = np.dot(b, p - a)
    closestPointOnLine = a + t*b
    return np.sqrt(np.sum((p - closestPointOnLine)**2))


@numba.njit()
def unravel_3d_index(index, shape):
    """
    Turn a 1D index into a 3D index.

    Equivalent to numpy.unravel_index specifically for
    3D; numpy's function isn't supported by numba, so I
    had to write my own.

    Parameters
    ----------
    index : int
        1D index for an array; for example, generated by
        `np.argmax(arr)`.

    shape : list[3]
        Size of each dimension of the array you want to index.

    Returns
    -------
    rIndex : list[3]
    """
    properIndex = np.zeros(3, dtype=np.int64)

    properIndex[0] = np.floor(index / (shape[1]*shape[2]));
    properIndex[1] = np.floor((index - properIndex[0] * shape[1]*shape[2]) / shape[2]);
    properIndex[2] = index - shape[2] * (properIndex[1] + shape[1] * properIndex[0]);

    return properIndex

def animateGeometries(geometries, outputFile='animation', rotationSpeed=5, maxFrames=None, fps=30, pointSize=7, lineWidth=5, zoom=0.8, backgroundColor=[1,1,1], crop=False, cropPadding=10, outputFormat='gif'):
    """
    Create an animation of a set of geometries rotating.

    Uses Open3D as the graphical framework.

    Parameters
    ----------
    geometries : list(open3d.geometry.Geometry)
        The list of geometries to draw, eg. point clouds, line sets, etc.

    outputFile : str
        The name of the animation file that will be saved. If it includes
        an (acceptable) extension, the output type will be automatically
        chosen; otherwise, the output format will be set by the
        kwarg `outputFormat`.

    rotationSpeed : int
        The speed of the rotation during the animation.

    maxFrames : int or None
        The total number of frames to save; if `None`, no limit will
        be applied.

        Note that I can't figure out how to exit a window automatically,
        so the window itself will continue existing until the user manually
        presses 'q', but no more than `maxFrames` will actually be saved.

    fps : int
        The frames per second of the output animation.

    pointSize : float
        The size of points for point clouds.

    lineWidth : float
        The width of lines for line sets (WIP).

    zoom : float
        The zoom setting for the view.

    backgroundColor : [float, float, float]
        The background color for the animation.

    crop : bool
        Whether to crop the animation to a minimal window size (plus
        padding) that only includes pixels that actually change
        during the animation.

    cropPadding : int
        The amount of padding to add around the animation if
        `crop=True`.

    outputFormat : 'gif' or 'mp4'
        The output format for the animation. Overridden if an extension is
        directly included in the file name (`outputFile`).
        
    """
    # Import these here since they are needed in the rest of this file
    # and this is sortof a debug method
    import cv2
    from PIL import Image
    import open3d as o3d

    ALLOWED_FORMATS = ['gif', 'mp4']

    # Try to infer the output format from the file name
    potentialExtension = outputFile.split('.')[-1].lower()
    if potentialExtension in ALLOWED_FORMATS:
        outputFormat = potentialExtension
        fullOutputPath = outputFile
    else:
        # Otherwise, we have to append whatever the provided extension is
        outputFormat = outputFormat.lower()

        # But first make sure the provided format is allowed
        if outputFormat not in ALLOWED_FORMATS:
            raise Exception(f'Invalid output format specified: {outputFormat}; available options are {ALLOWED_FORMATS}.')

        fullOutputPath = f'{outputFile}.{outputFormat}'

    images = []

    def update_view(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray(backgroundColor)

        vis.get_render_option().point_size = pointSize
        vis.get_render_option().line_width = lineWidth


        ctr = vis.get_view_control()
        ctr.rotate(rotationSpeed, 0)
        ctr.set_zoom(zoom)

        if maxFrames is None or len(images) < maxFrames:
            image = (np.array(vis.capture_screen_float_buffer(False))*255).astype(np.uint8)
            images.append((image).astype(np.uint8))

        return False

    o3d.visualization.draw_geometries_with_animation_callback(geometries,
                                                              update_view)

    if crop:
        # If we want to crop, we should see if there are any pixels that only
        # ever take on the value of the background color
        usedPixels = np.sum(np.array(images) - np.array(backgroundColor)*255, axis=0)
        usedPixels = np.sum(usedPixels**2, axis=-1)

        usedIndices = np.array(np.where(usedPixels > 1e-8))
        cropBounds = [max(np.min(usedIndices[0])-cropPadding, 0),
                      min(np.max(usedIndices[0])+cropPadding, images[0].shape[0]),
                      max(np.min(usedIndices[1])-cropPadding, 0),
                      min(np.max(usedIndices[1])+cropPadding, images[0].shape[1])]

    else:
        cropBounds = [0, images[0].shape[1], 0, images[0].shape[0]]

    for i in range(len(images)):
        images[i] = images[i][cropBounds[0]:cropBounds[1],cropBounds[2]:cropBounds[3]]

    if outputFormat == 'mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(fullOutputPath, fourcc, fps, (images[0].shape[1], images[0].shape[0]))

        for image in images:
            video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        video.release()

    if outputFormat == 'gif':
        for i in range(len(images)):
            images[i] = Image.fromarray(images[i])

        images[0].save(fullOutputPath, save_all=True, append_images=images[1:], loop=0, duration=int(1/fps*1e3))

