# Q1

# px = r * cos(φ)
# py = h
# pz = r * sin(φ)

# Q2

import numpy as np

def compute_homography(u1, u2):
  # Create the A matrix
  A = np.zeros((8, 8))
  for i in range(4):
    A[2*i] = [-u1[i][0], -u1[i][1], -1, 0, 0, 0, u1[i][0]*u2[i][0], u1[i][1]*u2[i][0]]
    A[2*i+1] = [0, 0, 0, -u1[i][0], -u1[i][1], -1, u1[i][0]*u2[i][1], u1[i][1]*u2[i][1]]

  # Compute the SVD of A
  U, S, V = np.linalg.svd(A)

  # The homography is the last column of V, reshaped into a 3x3 matrix
  H = V[-1, :].reshape((3, 3))

  return H

# Example input: 4 correspondences between the first and second images
u1 = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
u2 = [[x1', y1'], [x2', y2'], [x3', y3'], [x4', y4']]

# Compute the homography from the first to the second image
H = compute_homography(u1, u2)


# Q4

# φ = atan2(v, u)
# h = r

# # ----------------------------------
# import cv2
# import numpy as np

# Read in the images
# images = []
# for i in range(8):
#     img = cv2.imread('image{}.jpg'.format(i+1))
#     images.append(img)

# # Stitch the images together using SIFT
# stitcher = cv2.Stitcher.create(cv2.Stitcher_SIFT)
# status, panorama = stitcher.stitch(images)

# # Check if the stitching was successful
# if status == cv2.Stitcher_OK:
#     # Convert the panoramic image to cylindrical coordinates
#     height, width, _ = panorama.shape
#     focal_length = width / 2
#     panorama_cylinder = np.zeros((height, width, 3), np.uint8)
#     for u in range(width):
#         for v in range(height):
#             # Convert from pixel coordinates (u,v) to cylindrical coordinates (φ,h)
#             phi = (u - width/2) / focal_length
#             h = (v - height/2) / focal_length
#             # Map the point on the cylinder back onto the image plane
#             x = np.int(focal_length * np.tan(phi)) + width/2
#             y = h * focal_length + height/2
#             if x >= 0 and x < width and y >= 0 and y < height:
#                 panorama_cylinder[v,u] = panorama[y,x]

# # Save the panoramic image to a file
# cv2.imwrite('panorama.jpg', panorama_cylinder)


