import carla
import time
import math
import pygame
import numpy as np
import random
import queue
import time


# ==============================
# Configuration
# ==============================
HOST = 'localhost'
PORT = 2000
TM_PORT = 8000       # Traffic Manager port
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FOV = 110
VEHICLE_BP_FILTER = "vehicle.tesla.model3"
SPEED_PERCENTAGE = 50.0  # reduce speed to 50% of speed limit
SPEED = 20.0  # km/h

# ==============================
# Connect to CARLA
# ==============================
client = carla.Client(HOST, PORT)
client.set_timeout(30.0)
world = client.get_world()
blueprints = world.get_blueprint_library()

# ==============================
# Spawn vehicle
# ==============================
vehicle_bp = random.choice(blueprints.filter("vehicle.*"))
spawn_point = random.choice(world.get_map().get_spawn_points())
vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

if vehicle is None:
    raise RuntimeError("Failed to spawn vehicle!")

print(f"Spawned vehicle {vehicle.type_id} at {spawn_point.location}")

# ==============================
# Set up Traffic Manager autopilot
# ==============================
traffic_manager = client.get_trafficmanager(TM_PORT)
vehicle.set_autopilot(True, TM_PORT)

# Optional: customize driving behavior
traffic_manager.vehicle_percentage_speed_difference(vehicle, SPEED_PERCENTAGE)
traffic_manager.distance_to_leading_vehicle(vehicle, 5.0)  # meters
traffic_manager.auto_lane_change(vehicle, True)

print("Autopilot enabled via Traffic Manager")

# ==============================
# Attach RGB camera
# ==============================
camera_bp = blueprints.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', str(CAMERA_WIDTH))
camera_bp.set_attribute('image_size_y', str(CAMERA_HEIGHT))
camera_bp.set_attribute('fov', str(CAMERA_FOV))

camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))  # front roof
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# ==============================
# Thread-safe queue for images
# ==============================
frame_queue = queue.Queue(maxsize=1)

def camera_callback(image):
    """Callback from CARLA sensor thread"""
    if not frame_queue.full():
        frame_queue.put(image)

camera.listen(camera_callback)

# ==============================
# Pygame setup
# ==============================
pygame.init()
display = pygame.display.set_mode((CAMERA_WIDTH, CAMERA_HEIGHT))
pygame.display.set_caption("CARLA Camera View")
clock = pygame.time.Clock()

# ==============================
# Main loop
# ==============================
try:
    print("Vehicle driving with autopilot. Press ESC or close window to quit.")
    running = True
    while running:
        # Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # Get latest frame
        if not frame_queue.empty():
            image = frame_queue.get()
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((CAMERA_HEIGHT, CAMERA_WIDTH, 4))
            array = array[:, :, [2, 1, 0]]  # BGRA -> RGB
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            display.blit(surface, (0, 0))
            pygame.display.flip()

        clock.tick(30)  # limit to 30 FPS

except KeyboardInterrupt:
    pass

finally:
    # ==============================
    # Clean up
    # ==============================
    camera.stop()
    camera.destroy()
    vehicle.destroy()
    pygame.quit()
    print("Actors destroyed, exiting.")