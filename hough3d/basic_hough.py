import numpy as np
import numba

from .utils import reducedRepConversionMatrices, distancePointToLine, unravel_3d_index

@numba.njit(cache=True)
def hough3D(points, directionVectors, latticeSize=128, neighborDistance=0.01, minPointsPerLine=5):
    """
    Perform line detection on a 3D point cloud.

    This version of the function is follows the algorithm described in
    the original paper [1] (except for step 5):
        1. Have all points vote in the hough space.
        2. Take the highest voted cell as an initial guess for a new line.
        3. Find all points less than a certain distance to the initial line.
        4. Perform linear regression on the nearby points to refine the line.
        5. Update which points are nearby to the line based on the updated line (NEW).
        6. Have all of the points that are part of this line reverse their vote.
        7. Save the information about the detected line.
        8. Repeat steps 2-7 until there are not enough points left to form a line.

    I found that adding step 5 gave slightly more reliable detections, so
    I have included it. Note that you could take this further, and perform
    several steps of fitting and updating, but here we just do one.
    
    Parameters
    ----------
    points : numpy.ndarray[N,3]
        Points representing the point cloud.

    directionVectors : numpy.ndarray[M,3]
        Direction vectors represented the discretized directions for
        the lines. Normally generated with `genIcosahedron()`.

    latticeSize : int
        The number of discrete boxes to use for the intercept variables in
        hough space.

    neighborDistance : float
        The distance a point can be from the initial guess of a
        line for it to be included in the linear regression
        used to refine the line parameters. Given as fraction of
        the total system size (diagonal length of the bounding box).

    minPointsPerLine : int
        The minimum number of points that must be part of a line
        detection for that line to be included in the final result.

    Returns
    -------
    linePointArr : numpy.ndarray[K,2,3]
        The position vectors for the start and end of each
        line.

    """
    ##############################################
    #              Preparation steps
    ##############################################
    # Shift the mean of the point cloud to the origin
    pointCloudTranslation = np.zeros(3)
    for i in range(3):
        pointCloudTranslation[i] = np.mean(points[:,i])
        
    points = np.copy(points) - pointCloudTranslation
    
    # Compute the discrete direction vectors for our lines
    #directionVectors = genIcosahedron(directionGranularity)
    # TODO: Make this generation function numba compatable
    
    # Find the extents of the point cloud
    bounds = np.zeros((2, 3))
    for i in range(3):
        bounds[:,i] = [np.min(points[:,i]), np.max(points[:,i])]
    #bounds = np.array([ for i in range(3)]).T

    systemLengthScale = np.sqrt(np.sum((bounds[1] - bounds[0])**2))

    stepSize = systemLengthScale / latticeSize

    # Our hough space will be a three dimensional array, with the first two
    # dimensions representing the reduced coordinates x' and y' for the
    # intersection point of a line with a plane that passes through [0,0,0],
    # and the last dimension for the direction vector of the line. Altogether,
    # a point i,j,k in the hough space fully describes a line in the space
    # of the point cloud.
    houghSpaceSize = (latticeSize, latticeSize, len(directionVectors))

    ##############################################
    #    Initial transformation to hough space
    ##############################################
    # The actual hough space where points will be voting
    houghSpace = np.zeros(houghSpaceSize)

    # Now do the voting
    for k in range(len(directionVectors)):
        b = directionVectors[k]
        # We can compute a vector in advance that we just dot with
        # each point to find the x' and y'
        # See equation 2 in Dalitz et al. 2017
        reducedRepMultX, reducedRepMultY = reducedRepConversionMatrices(b)

        # I'm not 100% clear on why adding the system length scale is
        # necessary here, but otherwise most of our lattice represents
        # values of x' and y' that aren't relevant...
        xPrime = np.dot(reducedRepMultX, points.T) + systemLengthScale/2
        yPrime = np.dot(reducedRepMultY, points.T) + systemLengthScale/2

        # Now convert to an index, so we can find the corresponding
        # place on the hough space lattice.
        xIndex = np.floor(xPrime / stepSize).astype(np.int32)
        yIndex = np.floor(yPrime / stepSize).astype(np.int32)

        # Get rid of indices that are not on our lattice
        validIndices = np.where((xIndex < latticeSize) & (yIndex < latticeSize))[0]

        #print(validIndices)
        for j in validIndices:
            houghSpace[xIndex[j], yIndex[j], k] += 1

    ##############################################
    #    Iterative selection of lines
    ##############################################
    # Now that we have calulated our hough space, we need to identify
    # maxima in it to find the best lines

    # We do this by starting with the most voted point in hough space
    # and then "unvote" all of the points that are associated with that
    # line, and then repeat the process iteratively
    
    # Numba doesn't play well with empty arrays because
    # it can't infer the type of the data, so we just start
    # with an empty line (all zeros) and we will remove
    # it at the end (see return statement).
    # First dimension indexes the line, second the start or end point
    # of the line, and the final dimension is for the space we are in.
    linePointArr = np.zeros((1, 2, 3))

    while True:
        # The highest voted space
        maxIndex = np.argmax(houghSpace)
        # Now we have to unravel the index, since it is given as a 1D index
        # The function numpy.unravel_index() does this, but it is not supported
        # by numba
        maxIndex = unravel_3d_index(maxIndex, houghSpaceSize)

        # Find the information about the line that this index
        # represents in hough space
        xIndex, yIndex, bIndex = maxIndex

        # DEBUG
        #plt.imshow(houghSpace[:,:,bIndex])
        #plt.colorbar()
        #plt.show()

        ##############################################
        #    Convert from hough space to real space
        ##############################################
        # Converted from index to reduced coordinates
        xPrime = xIndex*stepSize - systemLengthScale/2
        yPrime = yIndex*stepSize - systemLengthScale/2

        # Calculate the anchor point on the line, which,
        # together with the direction vector, fully identifies
        # the line.
        b = directionVectors[bIndex]
        reducedRepMultX, reducedRepMultY = reducedRepConversionMatrices(b)
        # See equation 3 from Dalitz et al. 2017
        anchorPoint = xPrime*reducedRepMultX + yPrime*reducedRepMultY

        # Now, we need to refine the line using linear regression.
        # First, that means identifying which points are included in
        # this line.

        # Compute the line intersection of the given line and
        # a perpendicular one that passes through every point, and then
        # find the distance from intersection to the respective point.
        distancesToPoints = np.array([distancePointToLine(p, anchorPoint, b) for p in points])
        nearbyPointIndices = np.where(distancesToPoints < systemLengthScale*neighborDistance)[0]

        # Make sure we have enough points to form a line
        nearbyPoints = points[nearbyPointIndices]

        if len(nearbyPoints) < minPointsPerLine:
            break

        ##############################################
        #    Linear regression on nearby points
        ##############################################
        # Now we perform linear regression on those points to fit a line
        # and increase the accuracy of our detected line

        # We will use SVD to find the first principle component
        # of the local points, which is equivalent to a linear
        # fit.
        # To do this, we need our data to be zero-centered
        # (and we need the center of mass point anyway as our
        # anchor point)
        anchorPoint = np.zeros(3)
        for i in range(3):
            anchorPoint[i] = np.mean(nearbyPoints[:,i])

        # SVD on the mean-centered data.
        # The original paper did PCA instead of SVD, as they said it
        # was much faster, but I think SVD is fine here...
        uu, dd, vv = np.linalg.svd(nearbyPoints - anchorPoint)
        # Direction vector is the first principle component
        directionVector = vv[0]

        # Update which points are close to the line based on
        # the better fit
        # NOTE: this is different from the original paper, as they do not
        # perform this step. I find it helps, so I've included it.
        distancesToPoints = np.array([distancePointToLine(p, anchorPoint, directionVector) for p in points])
        nearbyPointIndices = np.where(distancesToPoints < systemLengthScale*neighborDistance)[0]
        nearbyPoints = points[nearbyPointIndices]

        if len(nearbyPoints) < minPointsPerLine:
            break

        for i in range(3):
            anchorPoint[i] = np.mean(nearbyPoints[:,i])
           
        # Compute the length of the line
        # The best way to do this is to compute the parametric representation
        # of every point in nearbyPoints. Then we can take the maximum and
        # minimum parametric values, to find how far the line extends in each direction
        tArr = np.dot(directionVector, (nearbyPoints - anchorPoint).T)
        lineStart = anchorPoint + np.max(tArr)*directionVector
        lineEnd = anchorPoint + np.min(tArr)*directionVector

        linePoints = np.zeros((2, 3))
        linePoints[0] = lineStart
        linePoints[1] = lineEnd

        # Save the information about this line
        linePointArr = np.concatenate((linePointArr, linePoints[None,:,:]))
        
        ##############################################
        #    Removing votes from used points
        ##############################################
        # Now, we want to have all of the points
        # involved in this fit to "unvote" in the hough space,
        # such that we can find the next best line
        for k in range(len(directionVectors)):
            b = directionVectors[k]
            # We can compute a vector in advance that we just dot with
            # each point to find the x' and y'
            # See equation 2 in Dalitz et al. 2017
            reducedRepMultX, reducedRepMultY = reducedRepConversionMatrices(b)
    
            # I'm not 100% clear on why adding the system length scale is
            # necessary here, but otherwise most of our lattice represents
            # values of x' and y' that aren't relevant...
            xPrime = np.dot(reducedRepMultX, nearbyPoints.T) + systemLengthScale/2
            yPrime = np.dot(reducedRepMultY, nearbyPoints.T) + systemLengthScale/2
            #xPrime = np.dot(reducedRepMultX, points.T) + systemLengthScale/2
            #yPrime = np.dot(reducedRepMultY, points.T) + systemLengthScale/2
    
            # Now convert to an index, so we can find the corresponding
            # place on the hough space lattice.
            xIndex = np.floor(xPrime / stepSize).astype(np.uint8)
            yIndex = np.floor(yPrime / stepSize).astype(np.uint8)
    
            # Get rid of indices that are not on our lattice
            validIndices = np.where((xIndex < latticeSize) & (yIndex < latticeSize))[0]
            
            #print(validIndices)
            for j in validIndices:
                houghSpace[xIndex[j], yIndex[j], k] -= 1

        # Remove the used points from the list
        keepIndices = np.array([i for i in range(len(points)) if i not in nearbyPointIndices], dtype=np.int64)
        points = points[keepIndices]
        
        ##############################################
        #    DEBUG
        ##############################################
        # Uncomment these to plot each line after it is detected.
        # Doesn't work with numba, so you'll have to remove the annotation.
        
        # import open3d as o3d
        
        # flattenedLinePoints = linePointArr.reshape((len(linePointArr)*2, 3))
        
        # lineSet = o3d.geometry.LineSet()
        # lineSet.points = o3d.utility.Vector3dVector(flattenedLinePoints)
        # lineSet.lines = o3d.utility.Vector2iVector([[i,i+1] for i in range(len(flattenedLinePoints))[::2]])
        
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(nearbyPoints)
        
        # o3d.visualization.draw_geometries([lineSet, pcd])
        ##############################################
        #    DEBUG
        ##############################################

    # Remove the first entry, since that was a dummy entry
    # And we have to translate the point could back to its original position
    return linePointArr[1:] + pointCloudTranslation
