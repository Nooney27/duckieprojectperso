import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import copy

map_image = cv2.imread(r'C:\Users\inesl\OneDrive\Bureau\ETH 11\Duckietown\project\duckieprojectperso\map_clean.png', cv2.IMREAD_GRAYSCALE)

_, valid_areas = cv2.threshold(map_image, 200, 255, cv2.THRESH_BINARY)


plt.imshow(valid_areas, cmap='gray')
plt.title('Valid Areas of the Map')
plt.show()


valid_area_percentage = np.sum(valid_areas == 255) / valid_areas.size * 100
print(f"Percentage of valid area: {valid_area_percentage:.2f}%")


class Particle:
    def __init__(self, x, y, w):
        self.state = np.array([x, y])
        self.weight = w

    def predict_with_odometry(self, distance):
        self.state = self.state + distance

    def predict_without_odometry(self, u_t):
        sample = np.random.normal(u_t, 20)
        self.state = self.state + sample
        
    def update(self, z_t):
        covariance_matrix = 150 * np.eye(2)
        likelihood = stats.multivariate_normal.pdf(z_t, self.state, covariance_matrix)
        self.weight = self.weight * likelihood


def initialize_particles(num_particles, valid_areas):
    height, width = valid_areas.shape
    particles = []

    while len(particles) < num_particles:
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        if valid_areas[y, x] == 255:
            particles.append(Particle(x, y, 1.0))  # Initialize with equal weight
    
    return particles

def predict_particles_random_movement(particles, valid_areas, step_size=210):
    height, width = valid_areas.shape
    for p in particles:
        movement = np.random.uniform(-step_size, step_size, size=2)
        p.predict_without_odometry(movement)
        p.state[0] = np.clip(p.state[0], 0, width - 1)
        p.state[1] = np.clip(p.state[1], 0, height - 1)
        while valid_areas[int(p.state[1]), int(p.state[0])] != 255:
            p.state = np.array([np.random.randint(0, width), np.random.randint(0, height)])

def resample_particles(particles):
    weights = np.array([p.weight for p in particles], dtype=np.float64)
    weights /= np.sum(weights)
    indices = np.random.choice(range(len(particles)), size=len(particles), p=weights)
    new_particles = [copy.deepcopy(particles[i]) for i in indices]
    for p in new_particles:
        p.weight = 1.0
    return new_particles

def estimate_position(particles):
    positions = np.array([p.state for p in particles])
    return np.mean(positions, axis=0)



num_particles = 100
particles = initialize_particles(num_particles, valid_areas)


plt.imshow(valid_areas, cmap='gray')
plt.scatter([p.state[0] for p in particles], [p.state[1] for p in particles], color='red', s=2)
plt.title('Initial Particle Distribution')
plt.show()


num_steps = 50
for _ in range(num_steps):
    predict_particles_random_movement(particles, valid_areas, step_size=210//2)
    particles = resample_particles(particles)


estimated_position = estimate_position(particles)
plt.imshow(valid_areas, cmap='gray')
plt.scatter([p.state[0] for p in particles], [p.state[1] for p in particles], color='red', s=2)
plt.scatter(estimated_position[0], estimated_position[1], color='blue', s=10, label='Estimated Position')
plt.legend()
plt.title('Updated Particle Distribution with Estimated Position')
plt.show()
print(estimated_position)