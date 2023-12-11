import cv2
import os

# Load the occupancy grid map image (Map)
map_image = cv2.imread('occupancy_grid.png', 0)  # Load the image as grayscale

# Define constants and parameters
grid_resolution = 0.1  # meters per pixel (adjust as needed)
map_height, map_width = map_image.shape
robot_pose = [map_width // 2, map_height // 2] 

# Directory containing robot perception images
image_dir = r'C:\Users\inesl\OneDrive\Bureau\ETH 11\Duckietown\project\Tour2'


# Function to find the robot's position in the map using template matching
def find_robot_position(map_image, perception_image):
    result = cv2.matchTemplate(map_image, perception_image, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)
    return max_loc

# Process each image in the directory
for filename in sorted(os.listdir(image_dir)):
    if filename.endswith('.png'):
        image_path = os.path.join(image_dir, filename)

        # Load the image from the robot's perception
        perception_image = cv2.imread(image_path, 0)

        # Find the robot's position in the map
        robot_position = find_robot_position(map_image, perception_image)

        # Update the robot's estimated pose (x, y) based on the result
        robot_pose = [robot_position[0], robot_position[1]]

# Visualize the map with the robot's estimated position
map_with_robot = map_image.copy()
cv2.circle(map_with_robot, (robot_pose[0], robot_pose[1]), 5, (0, 0, 255), -1)  # Red dot for the robot
resized_map_with_robot = cv2.resize(map_with_robot, (500, 500))
cv2.imshow('Map with Robot', resized_map_with_robot)
cv2.waitKey(0)  # Wait for a key press to close the window

# Close the OpenCV window
cv2.destroyAllWindows()
