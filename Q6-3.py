# Q3
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_camera_z_axis_3d(rotation_matrices):
    # Extract the camera Z axis from the rotation matrices
    camera_z_axis = []
    for rotation_matrix in rotation_matrices:
        camera_z_axis.append(rotation_matrix[:,2])
    
    # Convert the camera Z axis to a list of x, y, and z coordinates
    x = []
    y = []
    z = []
    for axis in camera_z_axis:
        x.append(axis[0])
        y.append(axis[1])
        z.append(axis[2])
    
    # Create a 3D scatter plot to visualize the camera Z axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

def stitch_images_cylindrical(images):
    # Warp the images onto a cylinder
    cylinder_images = []
    for image in images:
        height, width, _ = image.shape
        cylinder_image = cv2.warpPerspective(image, K, (width, height))
        cylinder_images.append(cylinder_image)

    # Blend the overlapping areas between images to create a seamless panoramic image
    panoramic_image = cylinder_images[0]
    for i in range(1, len(cylinder_images)):
        panoramic_image = cv2.addWeighted(panoramic_image, 1, cylinder_images[i], 1, 0)

    return panoramic_image

def compute_rotation_matrix_feature_matching(prev_image, curr_image):
    # Convert the images to grayscale
    prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)

    # Detect features in the images using SIFT
    sift = cv2.xfeatures2d.SIFT_create()
    prev_kp, prev_des = sift.detectAndCompute(prev_gray, None)
    curr_kp, curr_des = sift.detectAndCompute(curr_gray, None)

    # Match the features between the images
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(prev_des, curr_des, k=2)

    # Filter the matches using the Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Extract the matched keypoints
    prev_pts = np.float32([prev_kp[m.queryIdx].pt for m in good_matches])
    curr_pts = np.float32([curr_kp[m.trainIdx].pt for m in good_matches])

    # Compute the transformation between the images
    M, _ = cv2.findHomography(prev_pts, curr_pts, cv2.RANSAC, 5.0)

    # Extract the rotation matrix from the transformation matrix
    rotation_matrix = M[:3,:3]

    return rotation_matrix

def get_exif_orientation(image):
    exif_data = image._getexif()
    if exif_data is not None:
        orientation = exif_data[274]
        return orientation
    return None

# Load the images
images = []
for i in range(1, 9):
    image = cv2.imread("image" + str(i) + ".jpg")
    images.append(image)

# Compute the rotation matrices for each image
rotation_matrices = []
for i in range(1, len(images)):
    prev_image = images[i-1]
    curr_image = images[i]
    
    # Compute the rotation matrix for each image
    rotation_matrix = compute_rotation_matrix_feature_matching(prev_image, curr_image)
    rotation_matrices.append(rotation_matrix)

# Visualize the camera Z axis in 3D using the rotation matrices
visualize_camera_z_axis_3d(rotation_matrices)

# Stitch the images together using the cylindrical projection
panoramic_image = stitch_images_cylindrical(images)

# Save the panoramic image
cv2.imwrite("panoramic_image.jpg", panoramic_image)

