import os
import cv2
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle

def checkContinue():
  goOn = True
  # wait in ms
  key = cv2.waitKey(1)
  if( key & 0xFF == ord('q')):
    goOn = False
  return goOn


class IO:
  def __init__( self, dataDir="JPEG", fileTemplate= "img_{}.jpeg"):
    self.dir = dataDir
    self.fileTemplate = fileTemplate
    # End of the file sequence
    self.end = False
    self.fileName = "empty"
    self.imgRead = 0
    
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
    # plot
    self.nFigRow = nFigRow
    self.nFigCol = nFigCol
    self.fig = 0
    self.ax = 0
    self.prevImg = img

    self.driftMax = 15
    self.state = 0
    self.imgCount = 0
    self.imgMean = 0
    self.prevClusterList = []
    self.finalClusterList = []
    self.deadZones = []
    self.newDeadZones = []
    # For culumn=0
    self.diffImg = 0
    self.grpImg = 0
    self.selectedClusters = []
    self.newClusters = []

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

  def drawClusterBox( self, ax, finalCluster, color):
    n = len(finalCluster)
    for i in range(n):
      (id, mean, cov, ev, frame, selected) = finalCluster[i]
      xMin, xMax, yMin, yMax = frame
      xMin = max(xMin-1, 0)
      xMax = min(xMax+1, 639)
      yMin = max(yMin-1, 0)
      yMax = min(yMax+1, 479)    
      dx = xMax - xMin
      dy = yMax - yMin
      
      if selected == 1:
        # rect = patches.Rectangle((50, 100), 40, 30, linewidth=1, edgecolor='r', facecolor='none')
        rect = patches.Rectangle((xMin, yMin), dx, dy, linewidth=1, edgecolor=color, facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
      else:
        rect = patches.Rectangle((xMin, yMin), dx, dy, linewidth=1, edgecolor='0.7', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)        
    return

  def drawDeadZoneBox( self, ax, color):
    for dz in self.deadZones:
      (ID, frame) = dz
      xMin, xMax, yMin, yMax = frame
      xMin = max(xMin-1, 0)
      xMax = min(xMax+1, 639)
      yMin = max(yMin-1, 0)
      yMax = min(yMax+1, 479)    
      dx = xMax - xMin
      dy = yMax - yMin
      
      # rect = patches.Rectangle((50, 100), 40, 30, linewidth=1, edgecolor='r', facecolor='none')
      rect = patches.Rectangle((xMin, yMin), dx, dy, linewidth=1, edgecolor=color, facecolor='none')
      # Add the patch to the Axes
      ax.add_patch(rect)
    return

  def increaseFrame( self, frame, dxy=6):
    xMin, xMax, yMin, yMax = frame
    xMin = max(xMin-dxy, 0)
    xMax = min(xMax+dxy, 639)
    yMin = max(yMin-dxy, 0)
    yMax = min(yMax+dxy, 479)    
    return (xMin, xMax, yMin, yMax)

  def filter( self, src, value):
    ni = src.shape[0]
    nj = src.shape[1]
    dType = src.dtype
    dst = np.zeros( (ni, nj),dtype=dType)
    
    dst = np.where (( src > value ) , 255 , 0)  

    return dst

  def minDistance(self, cluster1, cluster2 ):
    distMin = 640
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
    
    return np.sqrt( distMin ) 
    
  def mergeCluster( self, clusters, i, j):
      
    # Merge j -> in i
    (imgID_i, clSize_i, mean_i, cov_i, ev_i, iMean_i, cluster_i) = clusters[i]
    (imgID_j, clSize_j, mean_j, cov_j, ev_j, iMean_j, cluster_j) = clusters[j]
    cluster_i.extend( cluster_j)
    np_cluster = np.array( cluster_i ).T
    mean = np.mean( np_cluster, axis=1)
    cov = np.cov( np_cluster )
    ev, vp = np.linalg.eig( cov )
    iMean = (clSize_i*iMean_i + clSize_j*iMean_j) / (clSize_i + clSize_j)
    clSize_i += clSize_j
    clusters[i] = (imgID_i, clSize_i, mean, cov, ev, iMean, cluster_i)
    del clusters[j]    
    return clusters

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
        dCoarse = self.frameDistance( frame_i, frame_j)
        print("fuseCloseClusters dCoarse({:d},{:d})={:.2f}".format(i, j, dCoarse))
        if dCoarse < 3.5 : 
          # refine distance
          d = self.minDistance(cluster_i, cluster_j)
          print("fuseCloseClusters d({:d},{:d})={:.2f}".format(i, j, d))
          if d < 3.5:
            # print("??? before i", clusterList[i])
            # print("??? before j", clusterList[j])
            clusterList = self.mergeCluster( clusterList, i, j)
            print("Fuse cluster {:d} & {:d}".format(i,j))
            # print("??? before i", clusterList[i])
            # Don't increment j
          else:
            j+= 1
          #
        #
        else:
          j+=1
      #
      i+=1
      
    return clusterList

  def clusterize(self, src, img, imgID):
    clusterList = []
    ni = src.shape[0]
    nj = src.shape[1]
    dType = src.dtype
    imgGrp = np.zeros( (ni, nj),dtype=dType)
    done = np.ones( (ni, nj),dtype=dType)
    idx = np.where( src > 0 )
    done[idx] = 0
    # print("idx ", idx)
    grp = 256
    print("-----Cluster List -----")
    while (np.sum(done) != ni*nj):
      # new group/cluster
      # grp -= 1
      idx = np.where(done == 0)
      # print("idx",idx)
      i = idx[0][0]
      j = idx[1][0]
      neigh = [(i,j)]
      cluster = []
      clusterForImg = []
      while( len(neigh) > 0 ):
        (i, j) = neigh.pop()
        done[i,j] = 1
        # imgGrp[i,j] = grp
        clusterForImg.append((i,j))  
        # Permute i, j because of how display imshow
        cluster.append((j,i))
        if  ((i-1) >= 0) and (done[i-1, j] == 0):
          neigh.append( (i-1,j))
        if ((i+1) < ni) and ( done[i+1, j] == 0 ):
          neigh.append( (i+1,j))
        if ((j-1) >= 0) and ( done[i, j-1] == 0 ):
          neigh.append( (i,j-1))
        if ((j+1) < nj) and  ( done[i, j+1] == 0 ):
          neigh.append( (i,j+1))
      # End of a cluster
      np_cluster = np.array( cluster ).T
      clSize = np_cluster.shape[1]
      # 
      #
      if len(cluster) > 9 and self.isNotInADeadZone(cluster, clSize):
        # Update image of groups
        # print(imgGrp.shape)
        # print(cluster[0], cluster[1])
        for l in clusterForImg :
         imgGrp[l] = grp
        print("Cluster", 256-grp)
        grp -= 1
        # compute cluster features
        # mean_x = np.mean( np_cluster, axis=0)
        mean = np.mean( np_cluster, axis=1)
        cov = np.cov( np_cluster )
        ev, vp = np.linalg.eig( cov ) 
        print("  np_cluster shape", np_cluster.shape)
        print("  cluster size", clSize)
        print("  mean", mean)
        print("  covar", cov)
        print("  diag", ev)
        print(img.shape)
        pixelList = np.array( clusterForImg )
        I = np.sum( img[pixelList[:,0], pixelList[:,1]] )
        iMean = 1.0*I/clSize
        print("  I, iMean", I, iMean)
        #for l in clusterForImg :
        #  I += img[clusterForImg[l][0], clusterForImg[l][1] ] 
        clusterList.append( (imgID, clSize, mean, cov, ev, iMean, cluster))
    print("nbr of Groups", 256 - grp)
    print("nbr of clusterList", len(clusterList))
    clusterList = self.fuseCloseClusters(clusterList )
    return imgGrp, clusterList
  
  def diffPrevImg(self, img, store=False):
    idx = np.where ( img > self.prevImg)
    diff = np.zeros( img.shape, dtype=img.dtype)
    diff[idx] = img[idx] - self.prevImg[idx]
    """
    diff = img - self.prevImg
    diff = np.where (( diff < 0 ) , 0 , diff) 
    """
    print("diffPrevImg> min/max={}, {}, mean={}".format(np.min( diff), np.max(diff), np.mean(diff)) ) 
    if store:
      self.prevImg = copy.deepcopy( img )
      # ??? self.prevImg = img
    return diff 

  def frameDistance(self, frame0, frame1):
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
    print("dx, dy, d", dx, dy, d)
    return d
    # return sign * np.sqrt( dx*dx+dy*dy)
    return d

  def frameDistance2(self, frame0, frame1):
    # print("frame0", frame0)
    # print("frame1", frame1)
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

  def frameSurface(self, frame):
    xMin, xMax, yMin, yMax = frame
    return (xMax-xMin)*(yMax-yMin)

  def imageMean( self, img):
    if (self.state == 0):
      self.imgMean = np.zeros( img.shape, dtype=float64)
      self.imgCount = 0
    diffImg = self.diffPrevImg(img, store=True)
    self.imgMean += diffImg  
    self.imgCount += 1
    #
    return

  def updateDeadZone(self, newDeadZones, curImgID, dt=6):
    finalDeadZones = []
    # Remove old zones
    for dz in self.deadZones:
      (ID, frame) = dz
      if ID >= (curImgID - dt):
        finalDeadZones.append( (ID, frame))
    # Add the new zones
    for dz in newDeadZones:
      (ID, frame) = dz
      finalDeadZones.append( (ID, frame))
    #
    self.deadZones = finalDeadZones
    #
    return

  def isNotInADeadZone(self, cluster, clSize, areaMinOfTheCluster=0.15):
    n = len(self.deadZones)
    nPixels = 0
    outDZ = True
    for k, dz in enumerate(self.deadZones):
      imgID, frame = dz
      (xMin, xMax, yMin, yMax) = frame
      iMin=640; iMax=0; jMin = 480; jMax = 0
      for c in cluster:
        (i,j) = c
        if i >= xMin and i <= xMax and j >= yMin and j <= yMax:
          nPixels += 1
        #
        iMin=min(iMin, i); iMax=max(iMax, i);
        jMin=min(jMin, j); jMax=max(jMax, j);
      #
      iMin=max(iMin-self.driftMax, 0); iMax=min(iMax+self.driftMax, 639);
      jMin=max(jMin-self.driftMax, 0); jMax=min(jMax+self.driftMax, 479);      
      if nPixels >= areaMinOfTheCluster * clSize:
        outDZ = False
        xMin = min(xMin, iMin); xMax = max(xMax, iMax);
        yMin = min(yMin, jMin); yMax = max(yMax, jMax);
        # update DeadZone
        # self.deadZones[k] = ( imgID, (xMin, xMax, yMin, yMax) )
        # break
    #  
    return outDZ

  def clusterFrame(self, cluster):
    n = len(cluster)
    pixelList = np.array( cluster )
    xMin =np.min( pixelList[:,0] )
    xMax = np.max( pixelList[:,0] )
    yMin =np.min( pixelList[:,1] )
    yMax = np.max( pixelList[:,1] )
    return (xMin, xMax, yMin, yMax)

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

    # Identify same cluster
    d2Max = driftMax * driftMax
    n0 = len( prevClusterList)
    n1 = len( clusterList)
    # remaining cluster
    I0 = np.zeros(n0, dtype=float)
    I1 = np.zeros(n1, dtype=float)
    drift0 = -1*np.ones(n0, dtype=int)
    drift1 = -1*np.ones(n1, dtype=int)

    rCluster = []
    # Used for display
    selectedClusters =[]
    newClusters =[]
    newDeadZone = []
    #
    for i0 in range(n0):
      (id0, clSize0, mean0, cov0, ev0, iMean0, cluster0) = prevClusterList[i0]
      frame0 = self.clusterFrame( cluster0)
      surf0 = self.frameSurface(frame0)
      for i1 in range(n1):
        (id1, clSize1, mean1, cov1, ev1, iMean1, cluster1) = clusterList[i1]
        frame1 = self.clusterFrame( cluster1)
        surf1 = self.frameSurface(frame1)
        """ an Intersection exist 
        iFrame = self.frameIntersection( frame0, frame1)
        iSurf = self.frameSurface(iFrame)
        minSurf = min(surf0, surf1)
        if (iSurf/minSurf) > 0.:
        """
        """
        # distance between the cluster means
        dx = mean1[0] - mean0[0]
        dy = mean1[1] - mean0[1]
        d2 = dx*dx + dy*dy
        if d2 < d2Max:
        """
        d = self.frameDistance( frame0, frame1)
        print("TS frame distance ({:d},{:d}) = {:.2f}".format(i0,i1, d))
        if (d < 1.5):
          # print("  (i0={:d}, i1={:d} same cluster d2={:.1f} d2max={:.1f}".format(i0, i1, d2, d2Max))
          # print("  (i0={:d}, i1={:d} same cluster inter/minSurface={:.1f} cutOff={:.1f}".format(i0, i1, iSurf/minSurf, 0.))
          print("TS  (i0={:d}, i1={:d} same cluster dist={:.1f} cutOff={:.1f}".format(i0, i1, d, 1.0))
          drift0[i0] = i1
          drift1[i1] = i0
          I0[i0] = iMean0
          I1[i1] = iMean1
        #
      #
    #
    # prevCluster added in the final list
    # remove from 
    # i.e :
    # 1-Cluster selection
    #  - no cluster drift
    #  - or cluster drift with iDensity0 > iDensity1
    #  -> they are removed from prevClusterList and added
    #     to finalList
    # 2-Cluster with no drift clusters
    #  -> they are removed from prevClusterList and added
    #     to finalList
    # 3- All other clusters in clusterList are move to prevClusterList
    
    # Build rCluster
    print("TS drift0", drift0)
    print("TS drift1", drift1)
    
    for i1 in range(n1):
      (id1, clSize1, mean1, cov1, ev1, iMean1, cluster1) = clusterList[i1]
      i0 = drift1[i1]
      if i0 < 0:
        # First time the cluster appear
        print("  i1={:d} appear for the firt time".format(i1))
        rCluster.append( clusterList[i1] )
        newClusters.append( (id1, mean1, cov1, ev1, self.clusterFrame(cluster1), 1) )
      elif I1[i1] > I0[i0]:
        # The cluster is more intense
        print("  i1={:d} more dense than i0={:d} -> i1 selected".format(i1, i0))
        frame = self.clusterFrame(cluster1)
        self.finalClusterList.append( (id1, mean1, cov1, ev1, frame) )  
        selectedClusters.append( (id1, mean1, cov1, ev1, frame, 1) )
        frame = self.increaseFrame(frame)
        newDeadZone.append( (id1, frame) )        
      else:
        # The cluster is less intense
        # Take the previous cluster
        print("  i0={:d} more dense than i1={:d} -> i0 selected".format(i0, i1))
        (id0, clSize0, mean0, cov0, ev0, iMean0, cluster0) = prevClusterList[i0]
        frame = self.clusterFrame(cluster0)
        self.finalClusterList.append( (id0, mean0, cov0, ev0, frame) )  
        selectedClusters.append( (id0, mean0, cov0, ev0, frame, 0) )
        # Take the frame "t+1" for the DeadZone 
        frame = self.clusterFrame(cluster1)          
        frame = self.increaseFrame(frame)
        newDeadZone.append( (id0, frame) )        
    #     
    for i0 in range(n0):
      (id0, clSize0, mean0, cov0, ev0, iMean0, cluster0) = prevClusterList[i0]
      if drift0[i0] < 0:
        print("  No cluster à t+1, i0={:d} selected".format(i0))
        frame = self.clusterFrame(cluster0)
        self.finalClusterList.append( (id0, mean0, cov0, ev0, frame) )  
        selectedClusters.append( (id0, mean0, cov0, ev0, frame, 0) )
        frame = self.increaseFrame(frame)
        newDeadZone.append( (id0, frame) )        
       
    return rCluster, selectedClusters, newClusters, newDeadZone


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
      self.drawClusterBox( self.ax[2,j], self.selectedClusters, 'r')
      self.drawClusterBox( self.ax[2,j], self.newClusters, 'g')
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
    binaryImg = self.filter( diffImg, 40)
    grpImg, clusterList = self.clusterize( binaryImg, diffImg, img_count)
    self.prevClusterList, selectedClusters, newClusters, self.newDeadZones = self.timeSelection( self.prevClusterList, clusterList, driftMax= self.driftMax)
    # self.ax[2,j].imshow( diffImg, cmap=plt.colormaps["jet"])
    if plot:
      self.ax[2,j+1].imshow( grpImg, cmap=plt.colormaps["jet"], origin='lower')
      self.drawClusterBox( self.ax[2,j+1], selectedClusters, 'r')
      self.drawClusterBox( self.ax[2,j+1], newClusters, 'g')
      self.drawDeadZoneBox( self.ax[2,j+1], 'tab:purple')
    # Save for colum 0
    self.diffImg = diffImg
    self.grpImg = grpImg
    self.selectedClusters = selectedClusters
    self.newClusters = newClusters
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
    
  clusters = pickle.load( open( "clusters.obj", "rb" ) )
  plotTracesDistribution( clusters )
  
  nextImage = True
  sumAcquisition=0.
  sumComputing=0.
  sumStore=0.
  sumTotal=0.  
  n=0
  nSkip = 0
  nSkip = 8
  nSkip = 1624

  nEnd = 100+nSkip
  nEnd = 10000+nSkip
  # Init: read the first image
  # io = IO("PNG", "img_{}.png")
  # io = IO("DACQ", "img_{}.jpeg")
  io = IO("Dacq-06Janv23", "img_{}.jpeg")
  img = io.nextRead()
  for i in range(nSkip):
    img = io.nextRead()
    n += 1
  #
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
  
    # Display in ms

    n += 1
    # Try to Read next image
    sRead = time.perf_counter_ns()
    img = io.nextRead()
    nextImage = nextImage and (not io.end)
    eRead = time.perf_counter_ns()
  #
  pickle.dump( process.finalClusterList, open( "clusters.obj", "wb" ) )
  clusters = pickle.load( open( "clusters.obj", "rb" ) )
  plotTracesDistribution( clusters )

  # Terminate
  cv2.destroyAllWindows()

