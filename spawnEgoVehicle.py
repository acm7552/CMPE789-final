import carla
import time
import math
#import pygame
import numpy as np
import random
import queue
import time
import os


# ==============================
# Configuration
# ==============================
HOST = 'localhost'
PORT = 2000
TM_PORT = 8000       # traffic Manager port
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FOV = 110
VEHICLE_BP_FILTER = "vehicle.tesla.model3"
SPEED_PERCENTAGE = 50.0  # reduce speed to 50% of speed limit
SPEED = 20.0  # km/h



CAPTURE_INTERVAL = 0.1  # seconds between frames 
DURATION = 90  # seconds capturing
OUTPUT_DIR = "output"

# setup output folders

rgb_dir = os.path.join(OUTPUT_DIR, "rgb")
seg_dir = os.path.join(OUTPUT_DIR, "seg")
seg_raw_dir = os.path.join(OUTPUT_DIR, "seg_raw")
depth_dir = os.path.join(OUTPUT_DIR, "depth")

for d in [rgb_dir, seg_dir, seg_raw_dir, depth_dir]:
    os.makedirs(d, exist_ok=True)


client = carla.Client(HOST, PORT)
client.set_timeout(60.0)
world = client.get_world()

settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = CAPTURE_INTERVAL
world.apply_settings(settings)

blueprints = world.get_blueprint_library()

# Spawn vehicle
vehicle_bp = random.choice(blueprints.filter(VEHICLE_BP_FILTER))
spawn_point = random.choice(world.get_map().get_spawn_points())
vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

if vehicle is None:
    raise RuntimeError("Failed to spawn vehicle!")

print(f"Spawned vehicle {vehicle.type_id} at {spawn_point.location}")


# Set up Traffic Manager autopilot
traffic_manager = client.get_trafficmanager(TM_PORT)
vehicle.set_autopilot(True, TM_PORT)

# Optional: customize driving behavior
traffic_manager.vehicle_percentage_speed_difference(vehicle, SPEED_PERCENTAGE)
traffic_manager.distance_to_leading_vehicle(vehicle, 5.0)  # meters
traffic_manager.auto_lane_change(vehicle, True)

print("Autopilot enabled via Traffic Manager")

# attach cameras

# rgb
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))

def make_camera(bp_name):
    bp = blueprints.find(bp_name)
    bp.set_attribute('image_size_x', str(CAMERA_WIDTH))
    bp.set_attribute('image_size_y', str(CAMERA_HEIGHT))
    bp.set_attribute('fov', str(CAMERA_FOV))
    bp.set_attribute('sensor_tick', f'{CAPTURE_INTERVAL}')
    return world.spawn_actor(bp, camera_transform, attach_to=vehicle)

rgb_cam = make_camera('sensor.camera.rgb')
seg_cam = make_camera('sensor.camera.semantic_segmentation')
seg_raw_cam = make_camera('sensor.camera.semantic_segmentation')
depth_cam = make_camera('sensor.camera.depth')

# thread-safe queues for images




# Thread-safe queues
rgb_q = queue.Queue()
seg_q = queue.Queue()
seg_raw_q = queue.Queue()
depth_q = queue.Queue()

rgb_cam.listen(lambda img: rgb_q.put(img))
seg_cam.listen(lambda img: seg_q.put(img))
seg_raw_cam.listen(lambda img: seg_raw_q.put(img))
depth_cam.listen(lambda img: depth_q.put(img))

print("Capturing synchronized frames...")

frames = int(DURATION / CAPTURE_INTERVAL)

try:
    for _ in range(frames):
        world.tick()   # <- lockstep tick

        rgb = rgb_q.get()
        seg = seg_q.get()
        seg_raw = seg_raw_q.get()
        depth = depth_q.get()

        frame = rgb.frame

        rgb.save_to_disk(os.path.join(rgb_dir, f"{frame:06d}.png"))
        seg.save_to_disk(os.path.join(seg_dir, f"{frame:06d}.png"),
                         carla.ColorConverter.CityScapesPalette)
        seg_raw.save_to_disk(os.path.join(seg_raw_dir, f"{frame:06d}.png"),
                         carla.ColorConverter.Raw)
        depth.save_to_disk(os.path.join(depth_dir, f"{frame:06d}.png"),
                           carla.ColorConverter.LogarithmicDepth)

finally:
    rgb_cam.stop()
    seg_cam.stop()
    depth_cam.stop()
    vehicle.destroy()

    # Restore world settings
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    print("Done.")


# rgb_cam.listen(lambda image: image.save_to_disk(os.path.join(rgb_dir, f"{image.frame:06d}.png")))
# seg_cam.listen(lambda image: image.save_to_disk(os.path.join(seg_dir, f"{image.frame:06d}.png"),
#                                                carla.ColorConverter.CityScapesPalette))
# depth_cam.listen(lambda image: image.save_to_disk(os.path.join(depth_dir, f"{image.frame:06d}.png"),
#                                                  carla.ColorConverter.LogarithmicDepth))


# print(f"Capturing for {DURATION} seconds...")

# try:
#     time.sleep(DURATION)

# finally:
#     print("Stopping and cleaning up...")
#     rgb_cam.stop()
#     seg_cam.stop()
#     depth_cam.stop()
#     rgb_cam.destroy()
#     seg_cam.destroy()
#     depth_cam.destroy()
#     vehicle.destroy()
#     print(f"Saved frames to '{OUTPUT_DIR}/'")



# # pygame setup
# pygame.init()
# display = pygame.display.set_mode((CAMERA_WIDTH, CAMERA_HEIGHT))
# pygame.display.set_caption("CARLA Camera View")
# clock = pygame.time.Clock()

# Main loop
# print("Capturing frames...")
# start_time = time.time()
# frame_id = 0

# try:
#     while time.time() - start_time < DURATION:
#         # Get latest images if available
#         if not rgb_q.empty():
#             img = rgb_q.get()
#             img.save_to_disk(os.path.join(rgb_dir, f"{frame_id:06d}.png"))
#         if not seg_q.empty():
#             img = seg_q.get()
#             img.convert(carla.ColorConverter.CityScapesPalette)
#             img.save_to_disk(os.path.join(seg_dir, f"{frame_id:06d}.png"))
#         if not depth_q.empty():
#             img = depth_q.get()
#             img.convert(carla.ColorConverter.LogarithmicDepth)
#             img.save_to_disk(os.path.join(depth_dir, f"{frame_id:06d}.png"))

#         frame_id += 1
#         time.sleep(CAPTURE_INTERVAL)

# finally:
#     print("Stopping and cleaning up...")
#     rgb_cam.stop()
#     seg_cam.stop()
#     depth_cam.stop()
#     rgb_cam.destroy()
#     seg_cam.destroy()
#     depth_cam.destroy()
#     vehicle.destroy()
#     print(f"Saved {frame_id} frames to '{OUTPUT_DIR}/'")