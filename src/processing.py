import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

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
    
  def read(self):
    img = np.zeros(0)
    self.fileName = "/".join( (self.dir, self.fileTemplate.format(n)) )
    isHere = os.path.isfile(self.fileName)
    if isHere:
      img = cv2.imread(self.fileName, cv2.IMREAD_GRAYSCALE)
    else:
      self.end = True
    #
    return img
      
  
# Invalid
"""
def process( img,  img_count):
  i = int( img_count / nFigRow)
  j = img_count % nFigRow
  print( i, j)
  fig, ax = plt.subplots(nrows=nFigRow, ncols=nFigCol, figsize=(15, 7))  
  ax[0,0].imshow(img)
  histogram = cv2.calcHist([img], [0], None, [256], [0, 256])
  ax[0,1].plot(histogram, color='k')
  plt.show()
  img_count +=1
  # if (img_count == (nFigRow*nFigCol-1) ):
  return
"""

class Processing:
  def __init__( self, nFigRow=2, nFigCol=2):
    # plot
    self.nFigRow = nFigRow
    self.nFigCol = nFigCol
    self.fig = 0
    self.ax = 0

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

  def filter( self, src, value):
    ni = src.shape[0]
    nj = src.shape[1]
    dType = src.dtype
    dst = np.zeros( (ni, nj),dtype=dType)
    
    dst = np.where (( src > value ) , 255 , 0)  

    return dst
     
  def process( self, img,  img_count):
    j = int( img_count % self.nFigCol)
    # j = img_count % nFigRow
    print(  j)
    if (j==0):
      self.fig, self.ax = plt.subplots(nrows=self.nFigRow, ncols=self.nFigCol, figsize=(15, 7))  
    self.ax[0,j].imshow( self.filter(img, 40), cmap=plt.colormaps["jet"] )
    """
    histogram = cv2.calcHist([img], [0], None, [256], [0, 256])
    self.ax[1,j].plot(histogram, color='k')
    """
    self.ax[1,j].imshow(self.filter( self.min(self.max(img)), 40), cmap=plt.colormaps["jet"])
    if (j == (self.nFigCol-1) ):
      plt.show()
    return

#
if __name__ == "__main__":
    
  nextImage = True
  sumAcquisition=0.
  sumComputing=0.
  sumStore=0.
  sumTotal=0.  
  n=0

  # Init: read the first image
  # io = IO("PNG", "img_{}.png")
  io = IO("JPEG", "img_{}.jpeg")
  img = io.read()
  #
  process = Processing( 2, 4)
  
  while( nextImage ):
    print("Image Read:", io.fileName , img.shape)
    process.process( img, n )
  
    # eStore = time.perf_counter_ns()
  
    # Continue or not
    # Wait a key stroke in ms
    nextImage = checkContinue()
  
    # Display in ms

    n += 1
    # Try to Read next image
    sRead = time.perf_counter_ns()
    img = io.read()
    eRead = time.perf_counter_ns()
  """  
  print("-----------------------------------------------------------")  
  print("Averages on {} images:".format(n))
  print("  Total time={:.2f}, dacq={:.2f}, processing={:.2f}, storage={:.2f} in ms".format(
         n, sumTotal/n, sumAcquisition/n, sumComputing/n, sumStore/n) )
  print("  Rate={} fps".format( (1000.0*n)/ sumTotal ))
  """
  # Terminate
  cv2.destroyAllWindows()

