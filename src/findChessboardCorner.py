import numpy as np
import cv2 as cv
import glob

#dimension of the chessboard
nx = 19 #number of chessboard corner in x 
ny = 19 #number of chessboard corner in y

#termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((nx*ny,3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
#print(objp)

# Arrays to store object points and image points from all the images
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image place

images = glob.glob('*.jpeg') #get the images path
#print(images)

for fname in images: #loop for each images
    img = cv.imread(fname) #load the image 
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) #convert an image from BGR (Blue Green Red) 2 (to) GRAY
    invert_gray = 255 - gray #invert the black&white image (help with detection of chessboard corner because background must be white)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(invert_gray, (nx,ny), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        #print("true")
        objpoints.append(objp) #save the corner locations
        #print(objpoints)

        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria) #Refines the corner locations
        imgpoints.append(corners2) #save the refined corners locations
        print(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (nx,ny), corners2, ret) #drawing the chessboard corners
        newfname = fname.replace('.jpeg', '_CBC.jpeg') #new file name CBC for ChessBoardCorners
        cv.imwrite(newfname, img) #saving the image'''

#find the camera matrix (mtx), the distortions coefficients (dist), the rotation vectors (rvecs) and translation verctors (tvecs)
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)  
print(dist)

#print(objpoints)
#print(imgpoints)