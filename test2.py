import cv2
import numpy as np

# Paths to the images
top_view_image_path = r'C:\Users\inesl\OneDrive\Bureau\ETH 11\Duckietown\project\iphone0.png'
ground_image_path = r'C:\Users\inesl\OneDrive\Bureau\ETH 11\Duckietown\project\test_localization.png'

# Load the images
top_view_image = cv2.imread(top_view_image_path)
ground_image = cv2.imread(ground_image_path)

# Check if the images have been loaded properly
if top_view_image is None or ground_image is None:
    print("Error: Could not load one or more images.")
else:
    print("Images loaded successfully.")

# Convert images to grayscale for feature extraction
top_view_gray = cv2.cvtColor(top_view_image, cv2.COLOR_BGR2GRAY)
ground_gray = cv2.cvtColor(ground_image, cv2.COLOR_BGR2GRAY)

# Use Canny edge detector to find edges in both images
edges_top_view = cv2.Canny(top_view_gray, 100, 200)
edges_ground = cv2.Canny(ground_gray, 100, 200)

# Compare the edges using matchShapes()
match_score = cv2.matchShapes(edges_top_view, edges_ground, cv2.CONTOURS_MATCH_I1, 0)

# Display the match score
print("Match score:", match_score)

# Find the keypoints and descriptors for the images
orb = cv2.ORB_create()
keypoints_top_view, descriptors_top_view = orb.detectAndCompute(top_view_gray, None)
keypoints_ground, descriptors_ground = orb.detectAndCompute(ground_gray, None)

# Match the keypoints using a brute-force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors_top_view, descriptors_ground)

# Sort the matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw the matches
match_image = cv2.drawMatches(top_view_gray, keypoints_top_view, ground_gray, keypoints_ground, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the edges and matches
cv2.imshow('Edges Top View', edges_top_view)
cv2.imshow('Edges Ground View', edges_ground)
cv2.imshow('Matches', match_image)
cv2.waitKey(0)  # Wait for a key press to exit
cv2.destroyAllWindows()
