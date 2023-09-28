import os
import cv2
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle

# Stop processing on 'q' key pressed
def checkContinue():
  goOn = True
  # wait in ms
  key = cv2.waitKey(1)
  if( key & 0xFF == ord('q')):
    goOn = False
  return goOn

# Image reader
class IO:
  def __init__( self, dataDir="JPEG", fileTemplate= "img_{}.jpeg"):
    self.dir = dataDir
    self.fileTemplate = fileTemplate
    # End of the file sequence
    self.end = False
    self.fileName = "empty"
    self.imgRead = 0
  #  
  # Read the next file 'img_{xxx+1}.jpeg
  def nextRead(self):
    img = np.zeros(0)
    self.fileName = "/".join( (self.dir, self.fileTemplate.format(self.imgRead)) )
    isHere = os.path.isfile(self.fileName)
    if isHere:
      img = cv2.imread(self.fileName, cv2.IMREAD_GRAYSCALE)
      self.imgRead += 1
    else:
      self.end = True
    #
    return img
  #
  # Read the kth file
  def read(self, k):
    img = np.zeros(0)
    self.fileName = "/".join( (self.dir, self.fileTemplate.format(k)) )
    isHere = os.path.isfile(self.fileName)
    if isHere:
      img = cv2.imread(self.fileName, cv2.IMREAD_GRAYSCALE)
    else:
      self.end = True
    #
    return img      
  
class Processing:
  def __init__( self, img, nFigRow=2, nFigCol=2):
    #
    #  PROCESSING PARAMETERS
    #
    #
    #  Building Clusters
    #
    # 
    # Threshold to binarize an image 
    self.pBinaryCutOff = 40
    # Max (Coarse) distance  between 2 bounding boxes
    # of 2 clusters to be candidate to merging
    self.pFuseFrameDistance = 3.5
    # Max distance  between 2 clusters
    # to merge them
    self.pFuseDistance = 3.5
    # Small clusters with an area less pMinClusterArea 
    # are ignored 
    self.pMinClusterArea = 9
    #
    # Dead Zone (DZ)
    # 
    # A cluster is considered falling in a Dead Zone (DZ) 
    # if the intersection area with the DZ, is greater than 
    # a fraction of the cluster area
    self.pDZIntersectionRatio = 0.15
    # Time duration of a Dead Zone (DZ)
    self.pDZTimeLimit = 6
    #
    #  Drift
    #
    # Max track/cluster drift (in pixels)
    self.pDriftMax = 15
    # Criterion to identify the same cluster @t-1 and t 
    self.pMaxDistanceForSameCluster = 1.5
    #
    #
    # Plot: number of plots per rows and columns
    self.nFigRow = nFigRow
    self.nFigCol = nFigCol
    self.fig = 0
    self.ax = 0

    # ??? self.state = 0
    # Image
    # Image counter
    self.imgCount = 0
    self.imgWidth = 640
    self.imgHeight = 480
    # Previous image
    self.prevImg = img
    # Image difference between the previous one
    self.diffImg = 0
    self.imgMean = 0
    # 
    # Clusters
    #
    # Groups to identify cluster in the image
    self.grpImg = 0
    # Clusters at t-1
    self.prevClusterList = []
    # Use for drawing
    # New clusters @ t
    self.drawNewClusters = []
    # Selected clusters among those @t-1 and @t
    self.drawSelectedClusters = []
    # List of all identified cluster during the procesing
    # (final result)
    self.finalClusterList = []
    #
    #  Dead zones
    #
    self.deadZones = []
    self.newDeadZones = []

  # Image filter taking the min of a kernel/window 3x3
  def min( self, src):
    ni = src.shape[0]
    nj = src.shape[1]
    print(src.dtype)
    dType = src.dtype
    dst = np.zeros( (ni, nj),dtype=dType)
    for i in range(1,ni-1):
      for j in range(1,nj-1):
        dst[i,j] = np.min( src[i-1:i+2, j-1:j+2])  
    return dst

  # Image filter taking the max of a kerne/window 3x3
  def max( self, src):
    ni = src.shape[0]
    nj = src.shape[1]
    print(src.dtype)
    dType = src.dtype
    dst = np.zeros( (ni, nj),dtype=dType)
    for i in range(1,ni-1):
      for j in range(1,nj-1):
        dst[i,j] = np.max( src[i-1:i+2, j-1:j+2])  
    return dst

  # Draw the bounding boxes of clusters
  # if 'selecled' field is True
  def drawClusterBox( self, ax, finalCluster, color):
    n = len(finalCluster)
    for i in range(n):
      (id, mean, cov, ev, frame, selected) = finalCluster[i]
      xMin, xMax, yMin, yMax = frame
      xMin = max(xMin-1, 0)
      xMax = min(xMax+1, self.imgWidth-1)
      yMin = max(yMin-1, 0)
      yMax = min(yMax+1, self.imgHeight-1)    
      dx = xMax - xMin
      dy = yMax - yMin
      
      if selected == 1:
        rect = patches.Rectangle((xMin, yMin), dx, dy, linewidth=1, edgecolor=color, facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
      else:
        #                                                                   ???
        rect = patches.Rectangle((xMin, yMin), dx, dy, linewidth=1, edgecolor='0.7', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)        
    return

  # Draw Dead Zone boxes
  def drawDeadZoneBox( self, ax, color):
    for dz in self.deadZones:
      (ID, frame) = dz
      xMin, xMax, yMin, yMax = frame
      xMin = max(xMin-1, 0)
      xMax = min(xMax+1, self.imgWidth-1)
      yMin = max(yMin-1, 0)
      yMax = min(yMax+1, self.imgHeight-1)    
      dx = xMax - xMin
      dy = yMax - yMin
      
      rect = patches.Rectangle((xMin, yMin), dx, dy, linewidth=1, edgecolor=color, facecolor='none')
      # Add the patch to the Axes
      ax.add_patch(rect)
    return

  # Increase a box of dxy on each side   
  def increaseFrame( self, frame, dxy=6):
    xMin, xMax, yMin, yMax = frame
    xMin = max(xMin-dxy, 0)
    xMax = min(xMax+dxy, self.imgWidth-1)
    yMin = max(yMin-dxy, 0)
    yMax = min(yMax+dxy, self.imgHeight-1)    
    return (xMin, xMax, yMin, yMax)

  # Binarisation of an image
  # - pixels > value are set to 255
  # - else pixels are set to zero
  def filterCutOff( self, src, value):
    ni = src.shape[0]
    nj = src.shape[1]
    dType = src.dtype
    dst = np.zeros( (ni, nj),dtype=dType)
    #
    dst = np.where (( src > value ) , 255 , 0)  

    return dst

  # Compute the minimal distance between two clusters
  def minDistance(self, cluster1, cluster2 ):
    distMin = self.imgWidth
    n1 = len(cluster1)
    n2 = len(cluster2)
    for i in range(n1):
      pi = cluster1[i]
      for j in range(n2):
        pj = cluster2[j]
        dx = pj[0] - pi[0]
        dy = pj[1] - pi[1]
        d = dx*dx + dy*dy
        distMin = min( distMin, d)
    #
    return np.sqrt( distMin ) 
  
  # Fuse two clusters in the cluster list
  def fuseTwoClusters( self, clusters, i, j):
      
    # Merge j -> in i
    (imgID_i, clSize_i, mean_i, cov_i, ev_i, iMean_i, cluster_i) = clusters[i]
    (imgID_j, clSize_j, mean_j, cov_j, ev_j, iMean_j, cluster_j) = clusters[j]
    # Add pixels of cluster j to the pixel list of cluster i
    cluster_i.extend( cluster_j)
    # Type cast to a numpy array
    np_cluster = np.array( cluster_i ).T
    # Compute the mean of the new cluster
    mean = np.mean( np_cluster, axis=1)
    # Take the new covariance
    cov = np.cov( np_cluster )
    # Compute the eigen values
    ev, vp = np.linalg.eig( cov )
    # ???
    iMean = (clSize_i*iMean_i + clSize_j*iMean_j) / (clSize_i + clSize_j)
    clSize_i += clSize_j
    # Update the new cluster i
    clusters[i] = (imgID_i, clSize_i, mean, cov, ev, iMean, cluster_i)
    del clusters[j]    
    return clusters

  # Merge close clusters in the cluster list
  # if the "coarse" distance betwee the 2 frames
  # is less than fuseFrameDistance
  def fuseCloseClusters(self, clusterList ):
    N = len(clusterList)
    i=0
    while i < len(clusterList):
      (imgID_i, clSize_i, mean_i, cov_i, ev_i, iMean_i, cluster_i) = clusterList[i]
      frame_i = self.clusterFrame(cluster_i)
      j=i+1
      while j <  len(clusterList):
        (imgID_j, clSize_j, mean_j, cov_j, ev_j, iMean_j, cluster_j) = clusterList[j]
        frame_j = self.clusterFrame(cluster_j)
        dCoarse = self.coarseFrameDistance( frame_i, frame_j)
        print("fuseCloseClusters dCoarse({:d},{:d})={:.2f}".format(i, j, dCoarse))
        if dCoarse < self.pFuseFrameDistance : 
          # Refine distance
          d = self.minDistance(cluster_i, cluster_j)
          print("fuseCloseClusters d({:d},{:d})={:.2f}".format(i, j, d))
          if d < self.pFuseDistance:
            # print("fuseCloseClusters> before i", clusterList[i])
            # print("fuseCloseClusters> before j", clusterList[j])
            clusterList = self.fuseTwoClusters( clusterList, i, j)
            print("Fuse cluster {:d} & {:d}".format(i,j))
            # print("fuseCloseClusters> before i", clusterList[i])
            # Don't increment j
          else:
            j+= 1
          #
        #
        else:
          j+=1
      #
      i+=1
    #
    return clusterList

  def clusterize(self, src, img, imgID):
    clusterList = []
    ni = src.shape[0]
    nj = src.shape[1]
    dType = src.dtype
    # Group map
    imgGrp = np.zeros( (ni, nj),dtype=dType)
    #
    # Pixels already processed 
    done = np.ones( (ni, nj),dtype=dType)
    idx = np.where( src > 0 )
    done[idx] = 0
    #
    grp = 256
    print("-----Cluster List -----")
    while (np.sum(done) != ni*nj):
      # new group/cluster
      # grp -= 1
      idx = np.where(done == 0)
      i = idx[0][0]
      j = idx[1][0]
      # Init neighbors list
      neigh = [(i,j)]
      cluster = []
      # Cluster map use for plots
      clusterForImg = []
      # Loop until empty neighbor list
      while( len(neigh) > 0 ):
        (i, j) = neigh.pop()
        done[i,j] = 1
        # Append the pixel in the cluster
        clusterForImg.append((i,j))  
        # Permute i, j because of how display imshow
        cluster.append((j,i))
        # Add neigbors of the current pixel in the list
        # if they hav'nt being processed
        if  ((i-1) >= 0) and (done[i-1, j] == 0):
          neigh.append( (i-1,j))
        if ((i+1) < ni) and ( done[i+1, j] == 0 ):
          neigh.append( (i+1,j))
        if ((j-1) >= 0) and ( done[i, j-1] == 0 ):
          neigh.append( (i,j-1))
        if ((j+1) < nj) and  ( done[i, j+1] == 0 ):
          neigh.append( (i,j+1))
      # End of a cluster
      # Type cast to numpy array
      np_cluster = np.array( cluster ).T
      clSize = np_cluster.shape[1]
      # 
      # Select only the clusters which its area 
      # is less than pMinClusterArea.
      # A cluster overlaping the dead zone list is ignored
      if len(cluster) > self.pMinClusterArea and self.isNotInADeadZone(cluster, clSize):
        # Update image of groups
        for l in clusterForImg :
          imgGrp[l] = grp
        # Next Group
        grp -= 1
        #
        # Compute cluster features
        #
        mean = np.mean( np_cluster, axis=1)
        cov = np.cov( np_cluster )
        ev, vp = np.linalg.eig( cov ) 
        """" Debug
        print("  np_cluster shape", np_cluster.shape)
        print("  cluster size", clSize)
        print("  mean", mean)
        print("  covar", cov)
        print("  diag", ev)
        print(img.shape)
        """
        pixelList = np.array( clusterForImg )
        I = np.sum( img[pixelList[:,0], pixelList[:,1]] )
        iMean = 1.0*I/clSize
        # ???? print("  I, iMean", I, iMean)
        clusterList.append( (imgID, clSize, mean, cov, ev, iMean, cluster))
    print("nbr of Groups", 256 - grp)
    print("nbr of clusterList", len(clusterList))
    # Fuse close clusters
    clusterList = self.fuseCloseClusters(clusterList )
    return imgGrp, clusterList
  
  # Positive Image difference with the previous image
  # diffImg = (img - prevImage) if (img > prevImage)
  #           else 0
  def diffPrevImg(self, img, store=False):
    idx = np.where ( img > self.prevImg)
    diff = np.zeros( img.shape, dtype=img.dtype)
    diff[idx] = img[idx] - self.prevImg[idx]

    print("diffPrevImg> min/max={}, {}, mean={}".format(np.min( diff), np.max(diff), np.mean(diff)) ) 
    if store:
      self.prevImg = copy.deepcopy( img )
    return diff 

  # Distance between 2 frames
  # Return max(dx, dy)
  # Return negative values if there 
  # is an intersection
  def coarseFrameDistance(self, frame0, frame1):
    # print("frame0", frame0)
    # print("frame1", frame1)
    xMin0, xMax0, yMin0, yMax0 = frame0
    xMin1, xMax1, yMin1, yMax1 = frame1
    xMax = max( xMin0, xMin1)
    xMin = min( xMax0, xMax1)
    yMax = max( yMin0, yMin1)
    yMin = min( yMax0, yMax1)
    dx = xMax - xMin
    dy = yMax - yMin
    d = max(dx,dy)
    # print("dx, dy, d", dx, dy, d)
    # return sign * np.sqrt( dx*dx+dy*dy)
    return d

  # Unused
  def frameDistance2(self, frame0, frame1):
    xMin0, xMax0, yMin0, yMax0 = frame0
    xMin1, xMax1, yMin1, yMax1 = frame1
    xMax = max( xMin0, xMin1)
    xMin = min( xMax0, xMax1)
    yMax = max( yMin0, yMin1)
    yMin = min( yMax0, yMax1)
    d = 0
    dx = xMax - xMin
    dy = yMax - yMin
    if (xMin > xMax) and (yMin > yMax):
      d = max(dx,dy)
    elif (xMin <= xMax) and (yMin > yMax):
      d = dx
    elif (xMin > xMax) and (yMin <= yMax):
      d = dy   
    elif (xMin <= xMax) and (yMin <= yMax):
      d = min( dx, dy)
    else:
      raise Exception("Imposible case") 
    print("xMin, xMax, yMin, yMax", xMin, xMax, yMin, yMax)
    print("dx, dy, d", dx, dy, d)
    # return sign * np.sqrt( dx*dx+dy*dy)
    return d

  # Return as a frame, the intersection between 2 frames
  def frameIntersection(self, frame0, frame1):
    xMin0, xMax0, yMin0, yMax0 = frame0
    xMin1, xMax1, yMin1, yMax1 = frame1
    xMin = max( xMin0, xMin1)
    xMax = min( xMax0, xMax1)
    yMin = max( yMin0, yMin1)
    yMax = min( yMax0, yMax1)
    if (xMin >= xMax):
       xMin=0; xMax=0;
    if (yMin >= yMax):
       yMin=0; yMax=0;
    return (xMin, xMax, yMin, yMax)

  # Return the frame area
  def frameSurface(self, frame):
    xMin, xMax, yMin, yMax = frame
    return (xMax-xMin)*(yMax-yMin)

  # Unused
  def imageMean( self, img):
    if (self.state == 0):
      self.imgMean = np.zeros( img.shape, dtype=float64)
      self.imgCount = 0
    diffImg = self.diffPrevImg(img, store=True)
    self.imgMean += diffImg  
    self.imgCount += 1
    #
    return

  # Add new Dead Zone to the current DZ list and 
  # remove old (dt >= pDZTimeLimit)
  def updateDeadZone(self, newDeadZones, curImgID):
    finalDeadZones = []
    # Remove old zones
    for dz in self.deadZones:
      (ID, frame) = dz
      if ID >= (curImgID - self.pDZTimeLimit):
        finalDeadZones.append( (ID, frame))
    # Add the new zones
    for dz in newDeadZones:
      (ID, frame) = dz
      finalDeadZones.append( (ID, frame))
    #
    self.deadZones = finalDeadZones
    #
    return

  # Return True if the cluster overlap is 
  # less than the intersection area is less than
  # a fraction (pDZIntersectionRatio) of its area
  def isNotInADeadZone(self, cluster, clArea ):
    imgWidth = self.imgWidth
    imgHeight = self.imgHeight
    n = len(self.deadZones)
    nPixels = 0
    outDZ = True
    for k, dz in enumerate(self.deadZones):
      imgID, frame = dz
      (xMin, xMax, yMin, yMax) = frame
      iMin=imgWidth; iMax=0; jMin = imgHeight; jMax = 0
      for c in cluster:
        (i,j) = c
        # count pixels in the DeadZone frame 
        if i >= xMin and i <= xMax and j >= yMin and j <= yMax:
          nPixels += 1
        # Define a frame
        iMin=min(iMin, i); iMax=max(iMax, i);
        jMin=min(jMin, j); jMax=max(jMax, j);
      #
      # Enlarge the frame with the drift
      iMin=max(iMin-self.pDriftMax, 0); iMax=min(iMax+self.pDriftMax, imgWidth-1);
      jMin=max(jMin-self.pDriftMax, 0); jMax=min(jMax+self.pDriftMax, imgHeight-1);      
      # The number of pixels in the Dead Zone
      # must be greater than a fraction of the cluster area
      if nPixels >= self.pDZIntersectionRatio * clArea:
        outDZ = False
        # Unused
        xMin = min(xMin, iMin); xMax = max(xMax, iMax);
        yMin = min(yMin, jMin); yMax = max(yMax, jMax);
    #  
    return outDZ

  # Return the Bounding Box of a cluster
  def clusterFrame(self, cluster):
    n = len(cluster)
    pixelList = np.array( cluster )
    xMin =np.min( pixelList[:,0] )
    xMax = np.max( pixelList[:,0] )
    yMin =np.min( pixelList[:,1] )
    yMax = np.max( pixelList[:,1] )
    return (xMin, xMax, yMin, yMax)

  #
  # The timeSelection selest the best tracks/cluster between two images
  # at t-1 and t. The criterion is the energy density of a cluster 
  # I(cluster) / Area(cluster at t-1, and t-1. cluster at t-1 and t 
  # are  the same if its drift (approximated with 
  # coarseFrameDistance( frame0, frame1) less than pMaxDistanceForSameCluster
  def timeSelection( self, prevClusterList, clusterList, driftMax=12.0):
    # drifMax in pixels
    print("------- timeSelection cluster@t-1 {:d} cluster@t {:d} driftMax={:4.1f} -------".format(
      len( prevClusterList), len( clusterList), driftMax))
    print("Cluster liste at t-1") 
    for k, c in enumerate(prevClusterList):
      # print(len(c))
      (id, clSize, mean, cov, ev, iMean, cl)= c
      print(" preCluster {:d} size={:d} mean={:.1f},{:.1f}".format(k, clSize, mean[0], mean[1]))
    print("Cluster liste at t")
    for k, c in enumerate(clusterList):
      # print(len(c))
      (id, clSize, mean, cov, ev, iMean, cl)= c
      print(" preCluster {:d} size={:d} mean={:.1f},{:.1f}".format(k, clSize, mean[0], mean[1]))
    #
    # Identify the same cluster at t-1, t
    #
    d2Max = driftMax * driftMax
    n0 = len( prevClusterList)
    n1 = len( clusterList)
    # I0, I1 clusters intensity
    I0 = np.zeros(n0, dtype=float)
    I1 = np.zeros(n1, dtype=float)
    drift0 = -1*np.ones(n0, dtype=int)
    drift1 = -1*np.ones(n1, dtype=int)

    # Clusters which must be considered @t+1 (next image)
    # to select the best one 
    keepClusters = []
    # Once the best cluster is found,
    # a new Dead Zone is created
    newDeadZone = []

    # Used for display 
    # Tags selected cluster and
    # the cluster appearing at t
    drawSelectedClusters =[]
    drawNewClusters =[]
    #
    # Identify the same clusters which have drifted
    #
    # Loop on clusters@t-1
    for i0 in range(n0):
      (id0, clSize0, mean0, cov0, ev0, iMean0, cluster0) = prevClusterList[i0]
      frame0 = self.clusterFrame( cluster0)
      surf0 = self.frameSurface(frame0)
      # Loop on clusters@t
      for i1 in range(n1):
        (id1, clSize1, mean1, cov1, ev1, iMean1, cluster1) = clusterList[i1]
        frame1 = self.clusterFrame( cluster1)
        surf1 = self.frameSurface(frame1)
        d = self.coarseFrameDistance( frame0, frame1)
        print("TS frame distance ({:d},{:d}) = {:.2f}".format(i0,i1, d))
        if (d < self.pMaxDistanceForSameCluster ):
          # print("  (i0={:d}, i1={:d} same cluster d2={:.1f} d2max={:.1f}".format(i0, i1, d2, d2Max))
          # print("  (i0={:d}, i1={:d} same cluster inter/minSurface={:.1f} cutOff={:.1f}".format(i0, i1, iSurf/minSurf, 0.))
          print("TS  (i0={:d}, i1={:d} same cluster dist={:.1f} cutOff={:.1f}".format(i0, i1, d, 1.0))
          # Build mapping 
          #  "drift0": index@t-1 -> index@t
          #  "drift1": index@t -> index@t-1
          # with their Intensity I0, I1
          drift0[i0] = i1
          drift1[i1] = i0
          I0[i0] = iMean0
          I1[i1] = iMean1
        #
      #
    #
    # Build keepClusters and update the final clusters 
    # to analyze their features 
    print("TS drift0", drift0)
    print("TS drift1", drift1)
    #
    # Select the best cluster among those @t-1 and @t
    # and create a dead zone. 
    # Store new clusters appearing @t
    
    # Cluster @t
    for i1 in range(n1):
      (id1, clSize1, mean1, cov1, ev1, iMean1, cluster1) = clusterList[i1]
      # Get the cluster index close to i1
      i0 = drift1[i1]
      if i0 < 0:
        # First time the cluster appear
        print("  i1={:d} appear for the firt time".format(i1))
        keepClusters.append( clusterList[i1] )
        drawNewClusters.append( (id1, mean1, cov1, ev1, self.clusterFrame(cluster1), 1) )
      elif I1[i1] > I0[i0]:
        # The cluster @t is more intense
        print("  i1={:d} more dense than i0={:d} -> i1 selected".format(i1, i0))
        frame = self.clusterFrame(cluster1)
        self.finalClusterList.append( (id1, mean1, cov1, ev1, frame) )  
        drawSelectedClusters.append( (id1, mean1, cov1, ev1, frame, 1) )
        # Add a new Dead Zone for cluster1
        frame = self.increaseFrame(frame)
        newDeadZone.append( (id1, frame) )        
      else:
        # The cluster @t-1 is more intense
        # Take the previous cluster
        print("  i0={:d} more dense than i1={:d} -> i0 selected".format(i0, i1))
        (id0, clSize0, mean0, cov0, ev0, iMean0, cluster0) = prevClusterList[i0]
        frame = self.clusterFrame(cluster0)
        self.finalClusterList.append( (id0, mean0, cov0, ev0, frame) )  
        drawSelectedClusters.append( (id0, mean0, cov0, ev0, frame, 0) )
        # Take the frame "t+1" for the DeadZone 
        frame = self.clusterFrame(cluster1)          
        frame = self.increaseFrame(frame)
        newDeadZone.append( (id0, frame) )        
    #     
    # Cluster @t-1 which are not identified @t
    for i0 in range(n0):
      (id0, clSize0, mean0, cov0, ev0, iMean0, cluster0) = prevClusterList[i0]
      if drift0[i0] < 0:
        print("  No cluster à t+1, i0={:d} selected".format(i0))
        frame = self.clusterFrame(cluster0)
        self.finalClusterList.append( (id0, mean0, cov0, ev0, frame) )  
        drawSelectedClusters.append( (id0, mean0, cov0, ev0, frame, 0) )
        frame = self.increaseFrame(frame)
        newDeadZone.append( (id0, frame) )        
       
    return keepClusters, drawSelectedClusters, drawNewClusters, newDeadZone


  def process( self, img,  img_count, plot=True):
    j = int( img_count % self.nFigCol)
    # j = img_count % nFigRow
    print(  j)
    if plot and (j==0):
      self.fig, self.ax = plt.subplots(nrows=self.nFigRow, ncols=self.nFigCol+1, figsize=(15, 10))
      self.ax[0,j].set_title("Image {:d}".format(img_count-1))
      self.ax[0,j].imshow( self.prevImg, cmap=plt.colormaps["gray"], origin='lower')
      self.ax[1,j].imshow( self.diffImg, cmap=plt.colormaps["gray"], origin='lower' )
      self.ax[2,j].imshow( self.grpImg, cmap=plt.colormaps["jet"], origin='lower')
      self.drawClusterBox( self.ax[2,j], self.drawSelectedClusters, 'r')
      self.drawClusterBox( self.ax[2,j], self.drawNewClusters, 'g')
      self.drawDeadZoneBox( self.ax[2,j], 'tab:purple')
    # self.ax[0,j].imshow( self.filter(img, 40), cmap=plt.colormaps["gray"] )
    self.updateDeadZone(self.newDeadZones, img_count)
    if plot:
      self.ax[0,j+1].set_title("Image {:d}".format(img_count))
      self.ax[0,j+1].imshow( img, cmap=plt.colormaps["gray"], origin='lower')
    diffImg = self.diffPrevImg(img, store=True)
    if plot:
      self.ax[1,j+1].imshow( diffImg, cmap=plt.colormaps["gray"], origin='lower' )
    """
    histogram = cv2.calcHist([img], [0], None, [256], [0, 256])
    self.ax[1,j].plot(histogram, color='k')
    """
    # self.ax[2,j].imshow(self.filter( self.min(self.max(img)), 40), cmap=plt.colormaps["jet"])
    # self.ax[2,j].imshow(self.prevImg, cmap=plt.colormaps["jet"])
    binaryImg = self.filterCutOff( diffImg, self.pBinaryCutOff)
    grpImg, clusterList = self.clusterize( binaryImg, diffImg, img_count)
    self.prevClusterList, drawSelectedClusters, drawNewClusters, self.newDeadZones = self.timeSelection( self.prevClusterList, clusterList, driftMax= self.pDriftMax)
    # self.ax[2,j].imshow( diffImg, cmap=plt.colormaps["jet"])
    if plot:
      self.ax[2,j+1].imshow( grpImg, cmap=plt.colormaps["jet"], origin='lower')
      self.drawClusterBox( self.ax[2,j+1], drawSelectedClusters, 'r')
      self.drawClusterBox( self.ax[2,j+1], drawNewClusters, 'g')
      self.drawDeadZoneBox( self.ax[2,j+1], 'tab:purple')
    # Save for colum 0
    self.diffImg = diffImg
    self.grpImg = grpImg
    self.drawSelectedClusters = drawSelectedClusters
    self.drawNewClusters = drawNewClusters
    if plot and (j == (self.nFigCol-1) ):
      plt.show()

    return

def plotTracesDistribution( clusters ):
  # Physical unit in cm : 22 cm -> 244 pixels
  pixelToCm = 22.0 / 224.0 
  n = len(clusters)
  nbTraces = 0
  ids = np.zeros(n,dtype=int)
  d = np.zeros(n,dtype=float)
  sigma1 = np.zeros(n,dtype=float)
  sigma2 = np.zeros(n,dtype=float)
  for k, cl in enumerate(clusters):
    (ID, mean, cov, ev, frame) = cl
    xMin, xMax, yMin, yMax = frame
    dx = xMax - xMin; dy = yMax - yMin
    dd = np.sqrt( dx*dx + dy*dy) * pixelToCm
    if (dd < 20.):
      ids[k] = ID
      d[k] = np.sqrt( dx*dx + dy*dy) * pixelToCm
      sigma1[k] = np.sqrt( np.max( ev ) ) * pixelToCm
      sigma2[k] = np.sqrt( np.min( ev ) ) * pixelToCm
      nbTraces += 1
  #
  kMin = np.argmin( d )
  kMax = np.argmax( d )  
  print("Cluster length min={:.1f}/{:d}, max={:.1f}/{:d}".format( d[kMin], ids[kMin], d[kMax], ids[kMax]))
  print("Number of traces ", nbTraces)
  #
  fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
  ax[0,0].hist( d, bins=100, log=True )
  ax[0,0].set_xlabel(r"$\rm{Longueur \; des \; traces \; (cm)}$")
  ax[0,0].set_ylabel(r"$\rm{Fréquences}$")

  ax[0,1].hist( sigma1, bins=100, log=True )
  ax[0,1].set_xlabel(r"$\sigma_1 \rm{(cm)}$")
  ax[0,1].set_ylabel(r"$\rm{Fréquences}$")
  ax[1,0].hist( sigma2, bins=100, log=True )
  ax[1,0].set_xlabel(r"$\sigma_2 \rm{(cm)}$")
  ax[1,0].set_ylabel(r"$\rm{Fréquences}$")
  #
  io = IO("Dacq-06Janv23", "img_{}.jpeg")
  img = io.read(ids[kMax])
  ax[1,1].imshow( img, cmap=plt.colormaps["gray"], origin='lower')
  ax[1,1].set_xlabel(r"$\rm{Image} \, (640x480) \rm{\; de \; la \; trace \; maximale \;} (l=16.4 \rm{cm})$")
  # 
  plt.tight_layout()
  plt.show()
  return
#
if __name__ == "__main__":
    
  # clusters = pickle.load( open( "clusters.obj.save", "rb" ) )
  # plotTracesDistribution( clusters )
  
  nextImage = True
  sumAcquisition=0.
  sumComputing=0.
  sumStore=0.
  sumTotal=0.  
  n=0
  # Skip images
  nSkip = 0
  nSkip = 8
  # Longest trace 
  # nSkip = 1624
  #
  # Stop preocessing
  nEnd = 100+nSkip
  nEnd = 10000+nSkip
  # Init: read the first image
  # io = IO("PNG", "img_{}.png")
  # io = IO("DACQ", "img_{}.jpeg")
  io = IO("../Dacq-06Janv23", "img_{}.jpeg")
  # Skip the images
  img = io.nextRead()
  for i in range(nSkip):
    img = io.nextRead()
    n += 1
  #
  # Init processing (ctor, prevImg, ...)
  process = Processing( img, 3, 4 )
  process.prevImg = copy.deepcopy( img )
  img = io.nextRead() 
  process.diffImg = process.diffPrevImg(img)
  process.grpImg = process.diffImg
  while( (n < nEnd) and nextImage ):
    print("Image Read:", io.fileName , img.shape)
    process.process( img, n, plot=True )
  
    # eStore = time.perf_counter_ns()
  
    # Continue or not
    # Wait a key stroke in ms
    nextImage = checkContinue()
  
    n += 1
    # Try to Read next image
    sRead = time.perf_counter_ns()
    img = io.nextRead()
    nextImage = nextImage and (not io.end)
    eRead = time.perf_counter_ns()
  #
  # Save cluster features (length, ...)
  pickle.dump( process.finalClusterList, open( "clusters.obj", "wb" ) )
  clusters = pickle.load( open( "clusters.obj", "rb" ) )
  plotTracesDistribution( clusters )

  # Terminate
  cv2.destroyAllWindows()

