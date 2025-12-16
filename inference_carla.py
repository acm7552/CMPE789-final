import carla
import numpy as np
import time
import pygame
import argparse
import torch
from PIL import Image

import seg_train_UNet_multiVehicle
import depth_train_UNet_multiVehicle


# ============================================================
# CONFIG
# ============================================================
IM_WIDTH = 640
IM_HEIGHT = 480

# ----- Define the 28-class CARLA palette -----
CARLA_PALETTE_28 = [
    0,0,0,      # 0 None
    70,70,70,   # 1 Buildings
    190,153,153,# 2 Fences
    72,0,90,    # 3 Other
    220,20,60,  # 4 Pedestrians
    153,153,153,# 5 Poles
    157,234,50, # 6 RoadLines
    128,64,128, # 7 Roads
    244,35,232, # 8 Sidewalks
    107,142,35, # 9 Vegetation
    0,0,142,    # 10 Vehicles
    102,102,156,# 11 Walls
    220,220,0,  # 12 TrafficSigns
    70,130,180, # 13 Sky
    81,0,81,    # 14 Ground
    150,100,100,# 15 Bridge
    230,150,140,# 16 RailTrack
    180,165,180,# 17 GuardRail
    250,170,30, # 18 TrafficLight
    110,190,160,# 19 Static
    170,120,50, # 20 Dynamic
    45,60,150,  # 21 Water
    145,170,100,# 22 Terrain
    255,0,0,    # 23 extra
    0,255,0,    # 24 extra
    0,0,255,    # 25 extra
    255,255,0,  # 26 extra
    255,0,255   # 27 extra
]
# pad to 768 entries for PIL palette requirement
CARLA_PALETTE_28 += [0] * (768 - len(CARLA_PALETTE_28))


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def carla_image_to_array(image):
    """Convert CARLA BGRA → RGB numpy array."""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))  # BGRA
    rgb = array[:, :, :3][:, :, ::-1]  # → RGB
    return rgb


def to_tensor(rgb):
    tensor = torch.from_numpy(rgb).permute(2,0,1).float() / 255.0
    return tensor.unsqueeze(0)  # NCHW


def make_seg_color(mask):
    """Convert mask (HxW) → RGB using palette."""
    img = Image.fromarray(mask.astype(np.uint8), mode="P")
    img.putpalette(CARLA_PALETTE_28)
    return np.array(img.convert("RGB"))


# ------ GLOBAL STORAGE ------
latest_segmentation_frame = None


def process_img(image):
    """Runs segmentation UNet and prepares an RGB numpy for pygame."""
    global latest_segmentation_frame, seg_unet, device

    # raw rgb
    rgb = carla_image_to_array(image)
    tensor = to_tensor(rgb).to(device)

    with torch.no_grad():
        out = seg_unet(tensor)

    mask = torch.argmax(out, dim=1)[0].cpu().numpy()
    colored = make_seg_color(mask)

    latest_segmentation_frame = colored  # saved for drawing


# ============================================================
# MAIN
# ============================================================

def main(args):
    global seg_unet, depth_unet, device

    # ---------------------
    # Load Models
    # ---------------------
    print("Loading segmentation model:", args.seg_unet)
    seg_unet.load_state_dict(torch.load(args.seg_unet, map_location=device))
    seg_unet.eval()

    print("Loading depth model:", args.depth_unet)
    depth_unet.load_state_dict(torch.load(args.depth_unet, map_location=device))
    depth_unet.eval()

    # ---------------------
    # CARLA
    # ---------------------
    client = carla.Client("localhost", 2000)
    client.set_timeout(60.0)
    world = client.get_world()

    # ---------------------
    # Pygame
    # ---------------------
    pygame.init()
    display = pygame.display.set_mode((IM_WIDTH, IM_HEIGHT))
    pygame.display.set_caption("CARLA Segmentation View")

    blueprint_library = world.get_blueprint_library()

    # Spawn vehicle
    bp = blueprint_library.filter("vehicle.tesla.model3")[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(bp, spawn_point)
    print("Vehicle spawned.")

    # Attach camera
    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(IM_WIDTH))
    camera_bp.set_attribute("image_size_y", str(IM_HEIGHT))
    camera_bp.set_attribute("fov", "110")

    cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, cam_transform, attach_to=vehicle)
    camera.listen(process_img)

    print("Manual drive: W A S D, SPACE brake, ESC quit")

    # Control loop
    try:
        while True:
            keys = pygame.key.get_pressed()
            control = carla.VehicleControl()

            if keys[pygame.K_w]:
                control.throttle = 0.6
            if keys[pygame.K_s]:
                control.brake = 0.3

            if keys[pygame.K_a]:
                control.steer = -0.5
            if keys[pygame.K_d]:
                control.steer = 0.5

            if keys[pygame.K_SPACE]:
                control.hand_brake = True

            vehicle.apply_control(control)

            # DRAW segmentation frame
            if latest_segmentation_frame is not None:
                surf = pygame.surfarray.make_surface(
                    latest_segmentation_frame.swapaxes(0,1)
                )
                display.blit(surf, (0,0))

            # Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    raise KeyboardInterrupt

            pygame.display.flip()
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        camera.stop()
        vehicle.destroy()
        pygame.quit()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seg_unet", default="unet_seg/unet_seg_29.pth")
    parser.add_argument("--depth_unet", default="unet_depth/unet_depth_best.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seg_unet = seg_train_UNet_multiVehicle.CARLA_UNet().to(device)
    depth_unet = depth_train_UNet_multiVehicle.CARLA_UNet().to(device)

    main(args)