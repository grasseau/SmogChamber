# Calibration FULL HD WEBCAM

## Specification FULL HD WEBCAM
WebCam HD USB
Ref: 900078
S/N PO152043
PORT Connect Europe
https://www.portdesigns.com/fr/1136-webcam-hd-1080.html
Technical Specifications : see file TechnicalSpecifications.jpg

## Reference for measuring distance from the webcam
The plane reference for measuring distance from the plane of the real image to the webcam, is the plane of the WebCam where the objetive protector is glued to the webcam. The error is about 2 mm

## Calibration images

There are two pictures taking at 50 cm and 70 cm distance between the calibration image and the webcam.

Each picture is taken ith two differents software
- PhotoBlooth in MacOs that change the number of pixel and it seems to process the image for optical optimisation
- ImageSnap command/software in the terminal window :
imagesnap -d JP001 WebCam-JP001-ImageSnap-50cm.jpg  
imagesnap -d JP001 WebCam-JP001-ImageSnap-70cm.jpg  


## Simple Analysis of the picture taken with ImageSnap (Gines, July 2023)
At 50 cm
Horizontal view 80+-1 cm, so horizontal angular view 2 x arctan (40./50.) = 77.3 degrees 
Vertical view 40+-1 cm, so horizontal angular view 2 x arctan(20./50.) = 43.6 degrees

Central picture (pixel 960,540) the calibration is 60+-2 pixel for 20 mm horizontally
Central picture (pixel 960,540) the calibration is 60+-2 pixel for 20 mm vertically

Left picture (pixel 210,540) the calibration is 42+-2 pixels for 20 mm horizontallly
Left picture (pixel 210,540) the calibration is 55+-2 pixels for 20 mm vertically

Right picture (pixel 1780,540) the calibration is 38+-2 pixels for 20 mm horizontally
Right picture (pixel 1780,540) the calibration is 52+-2 pixels for 20 mm vertically

Top picture (pixel 970,60) the calibration is 57+-2 pixels for 20 mm horizontally
Top picture (pixel 970,60) the calibration is 56+-2 pixels for 20 mm vertically

At 70 cm
Horizontal view 115+-3 cm, so horizontal angular view 2 x arctan (57.5./70.) = 78.8 degrees 
Vertical view 50+-2 cm, so horizontal angular view 2 x arctan(25./70.) = 39.3 degrees

612 160 mm