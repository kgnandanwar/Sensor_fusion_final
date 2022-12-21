import os
import cv2
import math
import numpy as np

from readim import *
from findmatch import *
from homo import *
from newframe import *
from stitchImage import *
   

if __name__ == "__main__":
    # Reading images.
    Images = ReadImage("InputImages/Field")
    
    BaseImage, _, _ = ProjectOntoCylinder(Images[0])
    for i in range(1, len(Images)):
        StitchedImage = StitchImages(BaseImage, Images[i])

        BaseImage = StitchedImage.copy()    

    cv2.imwrite("Stitched_Panorama_custom.png", BaseImage)
