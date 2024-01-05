# Cloud Chamber
- [The intership project 2022-2023](StageL3-ImageChambreBrouilard_2022-2023.pdf)
- [Technical description of the Cloud Chamber](SubatechCloudChamber_Doc1.pdf)
- [Trace visualization](SubatechCloudChamber_Doc2.pdf)

# Code(python) in src directory
- webcam_dacq.py: image acquisition from a web camera
- sPhone_dacq.py: image acquisition from a smartphone
- processing.py: basic image processing
- traceSimulation.py: trace simulation and trace length distribution
- TrackSimulaiont.py: simulates track in the cloud chamber

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
