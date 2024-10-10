import numpy as np

import numba

from scipy.signal import convolve


def courseGrainField(points, values=None, defaultValue=0, latticeSpacing=None, defaultLatticeSize=100, fixedBounds=None, kernel='gaussian', kernelSize=5, subsample=None, returnSpacing=False, returnCorner=False):
    """
    Course grains a collection of values at arbitrary points,
    into a discrete field.

    If `values=None`, course-grained field is the point density.

    Parameters
    ----------
    points : numpy.ndarray[N,d]
        Spatial positions of N points in d-dimensional space.

    values : numpy.ndarray[N,[k]] or func(points)->numpy.ndarray[N,[k]] or None
        Field values at each point. Can be k-dimensional vector,
        resulting in k course-grained fields.

        Can also be a (vectorized) function that returns a value given
        a collection of points. eg. neighbor counting function. This
        functionality is provided such that if the function is computationally
        expensive, eg. neighbor counting, the points can be subdivided into
        batches and the course grained fields can be summed at the end. This
        is a way to approximate the course grained field for a huge (>1e6)
        number of points, while still remaining computationally feasible.
        See `subsample`.

        If `None`, returned field will be the point density.

    defaultValue : float or numpy.ndarray[k]
        The default value of the course-grained field;
        probably `0` for most applications.

    latticeSpacing : float or None
        The spacing of lattice points for the course-grained field.

        If `None`, will be chosen such that each axis has
        `defaultLatticeSize` points.

    defaultLatticeSize : int
        The number of lattice points for the course grained field, assuming
        no explicit value for the lattice spacing is given (see `latticeSpacing`).

    fixedBounds : numpy.ndarray[d] or None
        The bounds of the field to define the discretized
        grid over. If None, will be calculated based on the
        extrema of the provided points.

    kernel : str or numpy.ndarray[A,A]
        The kernel to course-grain the field with. 'gaussian'
        option is implemented as default, but a custom matrix
        can be provided. If using default gaussian option,
        kernel size can be set with `kernelSize`.

    kernelSize : int
        The kernel size to use if `kernel='gaussian'`.
        If a custom kernel is provided, this has no effect.

    returnSpacing : bool

    returnCorner : bool
    """
    # TODO: Make sure this works for 1D data
    dim = np.shape(points)[-1] if len(np.shape(points)) > 1 else 1

    if dim == 1:
        points = np.array(points)[:,None]
    
    if not hasattr(fixedBounds, '__iter__'):
        occupiedVolumeBounds = np.array(list(zip(np.min(points, axis=0), np.max(points, axis=0))))
    else:
        occupiedVolumeBounds = np.array(fixedBounds)
    
    # Create a lattice with the selected scale for that cube
    if latticeSpacing is not None:
        spacing = latticeSpacing
        # We also have to correct the occupied volume bounds if we were provided with
        # a fixed set of bounds. Otherwise, we will end up with an extra bin at the
        # end
        if hasattr(fixedBounds, '__iter__'):
            occupiedVolumeBounds[:,1] -= spacing
    else:
        # Choose such that each axis has 100 lattice points (-1 because we will add one later)
        spacing = (occupiedVolumeBounds[:,1] - occupiedVolumeBounds[:,0]) / (defaultLatticeSize-1)

    # In the exceptional case that the data is given as a d dimensional array
    # but the data is actually d-1 dimensional (or d-2, etc.), we will have a value
    # of spacing for that dimension as 0, which will cause a divide by zero error
    # below. If this is the case, we only need a single entry in that dimension.
    if hasattr(spacing, '__iter__'):
        spacing[spacing == 0] = 1
    else:
        spacing = spacing if spacing != 0 else 1
    
    fieldDims = (np.ceil(1 + (occupiedVolumeBounds[:,1] - occupiedVolumeBounds[:,0])/(spacing))).astype(np.int64)

    latticePositions = np.floor((points - occupiedVolumeBounds[:,0])/(spacing)).astype(np.int64)

    # Check if an array of values was passed for each point
    # Otherwise we just have a scalar field (and we'll collapse
    # the last dimension later on).
    if hasattr(values, '__iter__'):
        k = np.shape(values)[-1]
        valArr = values
    else:
        k = 1
        valArr = np.ones((np.shape(points)[0], 1))

    fieldArr = np.zeros((*fieldDims, k))
    # Instead of actually applying a gaussian kernel now, which would be
    # very inefficient since we'd need to sum a potentially very large number
    # of k*d dimensional matrices (more or less), we instead just assign each
    # lattice point, then smooth over it after with the specified kernel.
    # Where this might cause issues:
    # - If the lattice spacing is too large, you will get some weird artifacts
    #   from this process. Though in that case, you'll get a ton of artifacts from
    #   elsewhere too, so just don't use too large a lattice spacing :)
    #print(tuple(latticePositions[0]))
    for i in range(np.shape(points)[0]):
        fieldArr[tuple(latticePositions[i])] += valArr[i]

    # Now smooth over the field
    if kernel == 'gaussian':
        gaussianBlurKernel = np.zeros(np.repeat(kernelSize, np.shape(points)[-1]))
        singleAxis = np.arange(kernelSize)
        kernelGrid = np.meshgrid(*np.repeat([singleAxis], np.shape(points)[-1], axis=0))
        #kernelGrid = np.meshgrid(singleAxis, singleAxis, singleAxis)
        # No 2 prefactor in the gaussian denominator because I want the kernel to
        # decay nearly to 0 at the corners
        kernelArr = np.exp(-np.sum([(kernelGrid[i] - (kernelSize-1)/2.)**2 for i in range(np.shape(points)[-1])], axis=0) / (kernelSize))
        # Now account for however many dimensions k we have
        #kernelArr = np.repeat([kernelArr] if k > 1 else kernelArr, k, axis=0)

    # Otherwise, we expect that kernel should already be passed as a
    # proper square d-dimensional matrix
    else:
        kernelArr = kernel

    # Perform a convolution of the field with our kernel
    # 'same' keeps the same bounds on the field, but might cause
    # some weird effects near the boundaries
    # Divide out the sum of the kernel to normalize
    transConvolution = np.zeros_like(fieldArr.T)

    for i in range(k):
        # Note that convolve(x, y) == convolve(x.T, y.T).T
        # We need this so we can go over our k axis
        transConvolution[i] = convolve(fieldArr.T[i], kernelArr.T, mode='same') / np.sum(kernelArr)

    convolution = transConvolution.T

    # If k == 1, collapse the extra dimension
    if k == 1:
        convolution = convolution[..., 0]
    
    returnResult = [convolution]

    if returnSpacing:
        returnResult += [spacing]

    if returnCorner:
        returnResult += [occupiedVolumeBounds[:,0]]

    return returnResult if len(returnResult) > 1 else convolution


@numba.njit()
def pathIntegralAlongField3D(field, path, latticeSpacing=np.array([1.,1.,1.]), fieldOffset=None):
    """
    numba-optimized function to extract values along a line through
    a 3D field. Can be easily modified to compute a path integral
    (noted in comments).

    Parameters
    ----------
    field : numpy.ndarray[N,M,P]
        Field over which to compute the path integral.

    path : numpy.ndarray[L,3]
        L ordered points representing a path
        through the field.

    latticeSpacing : numpy.ndarray[3]
        The lattice spacing for the discretized field;
        can be a single value for all dimensions, or different
        values for each dimension.

    fieldOffset : numpy.ndarray[3] or None
        The position of the bottom left corner
        of the discrete lattice on which the field exists.

    Returns
    -------
    pathCut : numpy.ndarray[L]

    """
    # This was written in a more general case,
    # so some of the code will look like it isn't specific
    # to 3D, but it has been made so here.
    d = field.ndim

    # Scale the path to have no units
    scaledPath = path.astype(np.float64)

    if fieldOffset is not None:
        scaledPath -= fieldOffset

    for i in range(d):
        scaledPath[:,i] /= latticeSpacing[i]

    nearbyIndices = []
    for i in range(len(scaledPath)):
        possibleIndices = []
        for j in range(d):
            belowIndex = np.floor(scaledPath[i,j])
            aboveIndex = np.ceil(scaledPath[i,j])
           
            # So this should just be [] but there is a chance
            # that this list could contain no elements, which numba does
            # not like at all. So we have to give some indication of what
            # type this list will hold, without actually giving it any elements.
            possibilities = [0 for _ in range(0)]

            # Make sure that the indices are valid
            if belowIndex >= 0 and belowIndex < field.shape[j]:
                possibilities += [belowIndex]
            if aboveIndex >= 0 and aboveIndex < field.shape[j]:
                possibilities += [aboveIndex]
            
            possibleIndices.append(np.array(possibilities))
            #possibleIndices.append(np.array([np.floor(scaledPath[i,j]), np.ceil(scaledPath[i,j])]))
        
        # Count up how many unique sets of indices we can create from the
        # possibilities for each axis
        #totalCombinations = int(np.nanprod(np.array([np.float64(len(p)) for p in possibleIndices])))
        totalCombinations = len(possibleIndices[0])*len(possibleIndices[1])*len(possibleIndices[2]) 

        # Have to manually index to make numba happy
        # TODO: Make sure this works if the z dimension is near the edge
        # I have tested it for x and y, but not the other one yet... if you have issues
        # around this area, it might be that.
        result = np.zeros((totalCombinations, d))
        result[:,0] = np.repeat(possibleIndices[0], len(possibleIndices[1])*len(possibleIndices[2]))
        result[:,1] = list(possibleIndices[1]) * len(possibleIndices[0])*len(possibleIndices[2])
        result[:,2] = list(np.repeat(possibleIndices[2], len(possibleIndices[1]))) * len(possibleIndices[0])
        
        #print(result)
        nearbyIndices.append(result)

    fieldValuesAlongPath = np.zeros(len(scaledPath))

    for i in range(len(scaledPath)):
        localPoints = nearbyIndices[i]
        if len(localPoints) == 0:
            continue

        # Compute distances to each nearby point
        # Add some tiny amount to avoid divide by zero issues
        localDistances = np.sqrt(np.sum((scaledPath[i] - localPoints)**2, axis=-1)) + 1e-8

        interpolationContributions = localDistances / np.sum(localDistances)

        #print(localPoints)
        # Have to do some weird indexing to make numba happy, but generally this is
        # just the dot product between interpolationContributions and field[all indices]
        for j in range(len(localPoints)):
            #index = (localPoints[j][0], localPoints[j][1], localPoints[j][2])
            fieldValuesAlongPath[i] += interpolationContributions[j] * field[int(localPoints[j][0]), int(localPoints[j][1]), int(localPoints[j][2])]

    return fieldValuesAlongPath
    # To get the actual path integral value, you can remove this
    # return statement

    # We need to weigh our numerical integration by the step size
    # We just to do a centered step scheme. ie. the interpolated value computed
    # above for point i "counts" for half of the path approaching point i, and
    # half of the path leaving point i. Thus, the first and last point are weighed
    # only half, since they don't have an incoming or outgoing path each.
    # We also have to scale back to the original lattice spacing.
    unscaledPath = scaledPath[1:] - scaledPath[:-1]
    for i in range(d):
        unscaledPath[:,i] *= latticeSpacing[i]

    pathSpacings = np.sqrt(np.sum((unscaledPath)**2, axis=-1))
    symSpacing = (pathSpacings[:-1] + pathSpacings[1:])/2
    symSpacing = np.concatenate((np.array([pathSpacings[0]/2]), symSpacing, np.array([pathSpacings[-1]/2])))

    pathIntegral = np.sum(symSpacing*fieldValuesAlongPath)

    return pathIntegral
