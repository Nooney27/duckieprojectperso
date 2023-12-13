import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
import scipy.stats as stats

class Particle:
    def __init__(self, x, y, w):
        self.state = np.array([x, y])
        self.weight = w

    def predict_with_odometry(self, left_dist, right_dist, wheelbase):
        center_dist = (left_dist + right_dist) / 2
        turning_angle = (left_dist - right_dist) / wheelbase if left_dist != right_dist else 0
        self.state[0] += center_dist * np.cos(turning_angle)
        self.state[1] += center_dist * np.sin(turning_angle)
        self.state[0] = np.clip(self.state[0], 0, valid_areas.shape[1] - 1)
        self.state[1] = np.clip(self.state[1], 0, valid_areas.shape[0] - 1)

    def update(self, z_t):
        covariance_matrix = 150 * np.eye(2)
        likelihood = stats.multivariate_normal.pdf(z_t, self.state, covariance_matrix)
        self.weight = self.weight * likelihood

def initialize_particles(num_particles, valid_areas):
    height, width = valid_areas.shape
    particles = []
    while len(particles) < num_particles:
        x, y = np.random.randint(0, width), np.random.randint(0, height)
        if valid_areas[y, x] == 255:
            particles.append(Particle(x, y, 1.0))
    return particles

def predict_particles_odometry(particles, left_dist, right_dist, wheelbase, valid_areas):
    for p in particles:
        p.predict_with_odometry(left_dist, right_dist, wheelbase)
        while valid_areas[int(p.state[1]), int(p.state[0])] != 255:
            p.state = np.array([np.random.randint(0, valid_areas.shape[1]), 
                                np.random.randint(0, valid_areas.shape[0])])

def resample_particles(particles):
    weights = np.array([p.weight for p in particles], dtype=np.float64)
    weights /= np.sum(weights)
    indices = np.random.choice(range(len(particles)), size=len(particles), p=weights)
    return [copy.deepcopy(particles[i]) for i in indices]

def estimate_position(particles):
    return np.mean(np.array([p.state for p in particles]), axis=0)

map_path = '/mnt/data/image.png'
map_image = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
_, valid_areas = cv2.threshold(map_image, 200, 255, cv2.THRESH_BINARY)
num_particles = 100
particles = initialize_particles(num_particles, valid_areas)
odometry_data = []  # Add your odometry readings here as tuples
wheelbase = 0.1
resolution = 210

for left_pixels, right_pixels in odometry_data:
    left_dist = left_pixels / resolution
    right_dist = right_pixels / resolution
    predict_particles_odometry(particles, left_dist, right_dist, wheelbase, valid_areas)
    particles = resample_particles(particles)

estimated_position = estimate_position(particles)
plt.imshow(valid_areas, cmap='gray')
plt.scatter([p.state[0] for p in particles], [p.state[1] for p in particles], color='red', s=2)
plt.scatter(estimated_position[0], estimated_position[1], color='blue', s=10, label='Estimated Position')
plt.legend()
plt.title('Updated Particle Distribution with Estimated Position')
plt.show()
print(estimated_position)



'''
# Initialize an empty list to hold the odometry data
odometry_data = []

# ... (rest of your setup code)

# Inside your format function or wherever you update odometry, append to odometry_data
def format(...):
    # ... (rest of your format function)
    
    # Append the left and right pixel distances to the odometry_data list
    odometry_data.append((left_pixels, right_pixels))
    
    # ... (rest of your format function)

# ... (rest of your Jupyter notebook code)

# At the end, when you want to stop collecting data, you would save the odometry_data
# For example, you can use numpy to save it to a file
np.save('odometry_data.npy', odometry_data)

THEN in filter.py
# Load the odometry data from the file saved by the Jupyter notebook
odometry_data = np.load('odometry_data.npy', allow_pickle=True)

# ... (rest of your particle filter code, using odometry_data)


'''