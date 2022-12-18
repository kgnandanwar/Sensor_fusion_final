import cv2
import numpy as np

# Read in the images
images = []
for i in range(8):
    img = cv2.imread('{}.jpg'.format(i+1))
    images.append(img)

# Stitch the images together using SIFT
stitcher = cv2.Stitcher.create(cv2.Stitcher_SIFT)
status, panorama = stitcher.stitch(images)

# Check if the stitching was successful
if status == cv2.Stitcher_OK:
    # Convert the panoramic image to cylindrical coordinates
    height, width, _ = panorama.shape
    focal_length = width / 2
    panorama_cylinder = np.zeros((height, width, 3), np.uint8)
    for u in range(width):
        for v in range(height):
            # Convert from pixel coordinates (u,v) to cylindrical coordinates (φ,h)
            phi = (u - width/2) / focal_length
            h = (v - height/2) / focal_length
            # Map the point on the cylinder back onto the image plane
            x = np.int(focal_length * np.tan(phi)) + width/2
            y = h * focal_length + height/2
            if x >= 0 and x < width and y >= 0 and y < height:
                panorama_cylinder[v,u] = panorama[y,x]

# Save the panoramic image to a file
cv2.imwrite('panorama.jpg', panorama_cylinder)
