import pybullet as p
import pybullet_data
import numpy as np
import time
import random

# --- PyBullet Setup ---
physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Load default assets
p.resetSimulation()
p.setGravity(0, 0, -9.8)

# Load Plane
plane_id = p.loadURDF("plane.urdf")

# Load Rover (Realistic Model)
rover_start_position = [0, 0, 0.2]
rover_orientation = p.getQuaternionFromEuler([0, 0, 0])
rover_id = p.loadURDF("husky/husky.urdf", rover_start_position, rover_orientation)

# Fire Locations
fire_positions = [
    [5, 5, 0.1],
    [-3, 4, 0.1],
    [4, -4, 0.1]
]

# Create Fire Particles
fire_particles = []
for fire_pos in fire_positions:
    for _ in range(8):  # Create particles per fire source
        p_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1)
        p_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[1, random.uniform(0.4, 0.7), 0, random.uniform(0.5, 1)])
        fire_particles.append(p.createMultiBody(0.01, p_shape, p_visual, basePosition=fire_pos))

# Add Static Obstacles
obstacles = [
    p.loadURDF("cube.urdf", [2, 0, 0.5], globalScaling=0.5),
    p.loadURDF("cube.urdf", [-2, -2, 0.5], globalScaling=0.7)
]

# Dynamic Obstacles
BOUNDARY_X = 5.0
BOUNDARY_Y = 5.0
obstacle_speed = 0.06
dynamic_obstacles = [p.loadURDF("sphere2.urdf", [random.uniform(-BOUNDARY_X, BOUNDARY_X), random.uniform(-BOUNDARY_Y, BOUNDARY_Y), 0.5]) for _ in range(5)]

# Fire Extinguishing Ball
ball_id = None  # Will be created later

# --- Rover Movement Variables ---
robot_speed = 1.00  # Movement speed
stop_distance = 3.0  # Distance to stop before fire

# --- Helper Functions ---
def detect_nearest_fire():
    """Find the nearest fire location."""
    rover_pos = np.array(p.getBasePositionAndOrientation(rover_id)[0])
    min_dist = float('inf')
    target_fire = None
    for fire_pos in fire_positions:
        dist = np.linalg.norm(np.array(fire_pos) - rover_pos)
        if dist < min_dist:
            min_dist = dist
            target_fire = fire_pos
    return target_fire, min_dist

def move_dynamic_obstacles():
    """Move obstacles randomly within boundaries."""
    for obstacle in dynamic_obstacles:
        position, _ = p.getBasePositionAndOrientation(obstacle)
        position = list(position)
        position[0] += random.uniform(-obstacle_speed, obstacle_speed)
        position[1] += random.uniform(-obstacle_speed, obstacle_speed)
        position[0] = np.clip(position[0], -BOUNDARY_X, BOUNDARY_X)
        position[1] = np.clip(position[1], -BOUNDARY_Y, BOUNDARY_Y)
        p.resetBasePositionAndOrientation(obstacle, position, [0, 0, 0, 1])

def detect_obstacles(robot_position):
    """Detect obstacles near the rover."""
    detected_obstacles = []
    for obstacle in dynamic_obstacles + obstacles:
        obs_pos = np.array(p.getBasePositionAndOrientation(obstacle)[0][:2])
        if np.linalg.norm(obs_pos - robot_position[:2]) < 1.0:  # Obstacle within 1m
            detected_obstacles.append(obs_pos)
    return detected_obstacles

def move_rover():
    """Move the rover toward the fire while avoiding obstacles."""
    global ball_id

    # Get the nearest fire location
    target_fire, distance_to_fire = detect_nearest_fire()
    if not target_fire:
        return

    rover_position = np.array(p.getBasePositionAndOrientation(rover_id)[0])
    direction = np.array(target_fire[:2]) - rover_position[:2]
    distance = np.linalg.norm(direction)

    if distance > stop_distance:  # Stop 3m before fire
        direction = direction / distance
        detected_obstacles = detect_obstacles(rover_position)

        # Obstacle Avoidance
        for obs in detected_obstacles:
            obs_direction = obs - rover_position[:2]
            obs_distance = np.linalg.norm(obs_direction)
            if obs_distance < 1.0:
                avoidance_direction = np.array([-obs_direction[1], obs_direction[0]])
                avoidance_direction /= np.linalg.norm(avoidance_direction)
                direction += avoidance_direction * 0.3

        direction = direction / np.linalg.norm(direction)
        rover_position[:2] += direction * robot_speed
        p.resetBasePositionAndOrientation(rover_id, rover_position, rover_orientation)

    else:
        print("Stopping 3m before fire!")
        if ball_id is None:
            launch_extinguishing_ball(target_fire)

def launch_extinguishing_ball(target_fire):
    """Launch a ball to extinguish the fire."""
    global ball_id
    rover_position = np.array(p.getBasePositionAndOrientation(rover_id)[0])
    ball_start = rover_position + np.array([0, 0, 0.5])  # Launch from rover

    # Create Ball
    ball_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.2)
    ball_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=[0, 0, 1, 1])  # Blue ball
    ball_id = p.createMultiBody(0.5, ball_shape, ball_visual, ball_start)

    # Calculate Direction & Apply Force
    direction = np.array(target_fire) - ball_start
    direction = direction / np.linalg.norm(direction) * 5  # Scale velocity
    p.applyExternalForce(ball_id, -1, direction, ball_start, p.WORLD_FRAME)

def update_fire_visuals():
    """Simulate fire with animated debug lines."""
    for fire_pos in fire_positions:
        for _ in range(10):
            offset = np.random.uniform(-0.3, 0.3, size=2)
            p.addUserDebugLine(
                lineFromXYZ=[fire_pos[0] + offset[0], fire_pos[1] + offset[1], fire_pos[2]],
                lineToXYZ=[fire_pos[0] + offset[0], fire_pos[1] + offset[1], fire_pos[2] + 1],
                lineColorRGB=[1, 0.5, 0],
                lifeTime=0.2
            )

# --- Main Simulation Loop ---
try:
    while True:
        move_dynamic_obstacles()
        move_rover()
        update_fire_visuals()
        p.stepSimulation()
        time.sleep(1. / 240.)
finally:
    p.disconnect()
