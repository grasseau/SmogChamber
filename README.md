# Cloud Chamber
- [The intership project 2023-2024](StageL3-ImageChambreBrouilard_2023-2024.pdf)
- [Technical description of the Cloud Chamber](SubatechCloudChamber_Doc1.pdf)
- [Trace visualization](SubatechCloudChamber_Doc2.pdf)

# Software: code(python3) in src directory
- src/webcam_dacq.py: image acquisition from a web camera (to be used in specific laptop for image data taking)

- src/two_webcam_dacq.py: image acquisition from two web cameras (to be used in specific laptop for image data taking) ir order to estimate the thickness of the active volume of the cloud chamber

- src/sPhone_dacq.py: image acquisition from a smartphone

- src/ processing.py: basic image processing (old Gilles G development)

- sec/ cloudChamberCommonCode.py : common code share by other py files

- src/findChessboardCorner.py: Code to find the corner of an ideal chess board and to fin the camera parameter for aberration correction 

- src/chessBoard_CameraCalibrationProcess.py : Code to correct for aberration of the webcamera

- src/filteringProcess.py : 
--- Definition of the images to be analyzed : 

--- Creation of dynamic (about each minute) associated background images (threshold, step and period : bck_img_)

--- Selection of the fiducial area, and threshold for binarisation

--- Creation of image to background difference (diff_img)

--- Creation of the binary image (bina_img)

--- Creation of a filtered (without small cluster) binary image (filt_img)

- src/rawClusteringProcess.py
--- Generation of the Raw Cluster Data : Dict with image number as key and a list of cluster per image

--- Structure of the cluster array:
clusterList.append((iImage, iCluster,  mean[0], mean[1], theta, sigmaLong, sigmaShort, clSize, extremeHighA, extremeHighB, extremeLowA, extremeLowB, mergingStatus, cluster))
cluster[0] = iImage : Image number. Same as key of the Dict containing the cluster list
cluster[1] = iCuster : number of the cluster in the image 
cluster[2] = mean[0] : x position in pixel of the center of the cluster    
cluster[3] = mean[1] : x position in pixel of the center of the cluster
cluster[4] = theta : angle of the direction of the largest axis of the cluster
cluster[5] = sigmaLong : Sqr(12) time the the RMS/1 in the axis of the larger eigenvalue of the cluster sigma matrix     
cluster[6] = sigmaShort : Sqr(12) time the the RMS/1 in the axis of the larger eigenvalue of the cluster sigma matrix     
cluster[7] = clSize : number of pixel in the cluster
cluster[8] = extremeHighA : 2D point in pixel of the cluster extreme
cluster[9] = extremeHighB : 2D point in pixel of the cluster extreme
cluster[10] = extremeLowA : 2D point in pixel of the cluster extreme
cluster[11] = extremeLowBB : 2D point in pixel of the cluster extreme
cluster[12] = MergingStaus : 0 or 1
cluster[13] = list of 2d pixel values members of the cluster

--- Creation of image with raw cluster  ellipses in red (clus_filt_img_) and long sqrt(12)x RMS line in green

--- Generation of control plots : 

--- output data rawClusteringData.dat, clus_filt_img* and rawClustering_ControlPlots.pdf

- src/mergingFragmentedClusterProcess.py
--- merging fragmented good clusters

--- the decision for merging is base in 3 parameters : distance of cluster point to the cluster line, relative angle between the two cluster and the relative distance between the two cluster

--- output data mergingFragmentedClusterData.dat, merg_clus_filt_img and mergingFragmentedCluster_ControlPlots.pdf

-- src/removingCorrelatedClusterProcess.py

--- removing cluster that are already present in the previous et following image to avoid double counting

--- out put data is MergedNoncorrelatedClusterData.dat

- src/distributionProcess.py
--- plots with good and not merged clusters in a fiducial volume

**Analysis on June 23 2024**
Simulation of 10^6 222Rn tracks in a box of 100 x 250 x 250 mm^2 with a active height of 5 mm
The total volume of the box in the simulation is 0.00625 m^2
So the total number of tracks in a m^3 is 1,6e8 = 1e6/0.00625 
The simulation shows that 2786 tracks decays in the active volume : 5 x 250 x 250 mm^2
The efficiency is then 1.74e-5 = 2786 / 1.6e8

In the experimental data on January 2024, one has identified about 1405 tracks in a gaussian peak about 39.4 pixel (one pixel is about 1 mm) and a width of 18.5%
The data taking took place on January 22 2025 between 11:40 and 12:00 am, so 20 minutes = 1200 s
The rate is 1.17 = 1405/1200 222Rn tracks in the cloud chamber per second

The total area is 325 x 150 pixel. If 1 pixel = 1 mm, the fiducial area in the simulation is more or less (within 20%) than in the real data taking. Let's consider the same.

Therefore 1.17 220Rn tracks per second, represents about 1,17 / 1.74e-5 = 67,2 kBq / m^2 of 222Rn

**Analysis on September 25 2024**
In the experimental data on January 2024, one has identified about 1611 tracks in a gaussian peak about 38.4 pixel (one pixel is about 1 mm) and a width of 19.9%
The data taking took place on January 22 2025 between 11:40 and 12:00 am, so 20 minutes = 1200 s
The rate is 1.34 = 1611/1200 222Rn tracks in the cloud chamber per second

The total fiducial area is 170 mm x 360 mm = 17 x 36 cm^2 = 612 cm^2









# Other files
- Data inputs: data acquisitions are image files (jpeg). There are 
  available in the Subatech file server.
- clusters.obj: file produces at the end of the processing/clustering
  to plot the length distribution. 
  - RANGE_3D_5489keV_N500.txt, RANGE_3D_6003keV_N500.txt, RANGE_3D_7687keV_N500.txt  event by event simulation of alpha particles in air for 5.5, 6.0 and 7.7 MeV from SRIM programme. Values provided graciously by Vinent METIVIER, Subatech
- Alpha_dans_air.pdf from SRIM programme. Values provided graciously by Vinent METIVIER, Subatech
-  AlphaRange_DryAir.dat and ProtonRange_DryAir.dat
Projected Range for protons/alpha in dry air from
https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html
https://physics.nist.gov/PhysRefData/Star/Text/ASTAR.html
these values are a good estimation for alpha particles in the range 1-10 MeV
