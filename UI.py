#!/usr/bin/env python

"""
InterFuser Real-time Monitoring UI - REDESIGNED VERSION
Professional UI matching the reference design
"""

import sys
import os
import glob
import math
import time
from collections import deque

# Project root directory (where UI.py lives)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Add CARLA Python API to path
try:
    sys.path.append(glob.glob(os.path.join(PROJECT_ROOT, 'CARLA_0.9.16', 'PythonAPI', 'carla', 'dist', 'carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64')))[0])
except IndexError:
    pass

# Add CARLA agents to path
sys.path.append(os.path.join(PROJECT_ROOT, 'CARLA_0.9.16', 'PythonAPI', 'carla'))

# Add InterFuser to path (insert at beginning to prioritize custom timm modules)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'interfuser_core'))

import carla
import pygame
import numpy as np
import cv2
import random

try:
    import queue
except ImportError:
    import Queue as queue

import torch
from timm.models.factory import create_model
from timm.models.helpers import load_checkpoint
from timm.data.transforms_carla_factory import create_carla_rgb_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.carla_dataset import lidar_to_histogram_features


# ============================================================================
# Control Parameters (from InterFuser paper)
# ============================================================================

TARGET_SPEED = 20.0  # km/h - default cruising speed
MAX_SPEED = 40.0     # km/h - absolute maximum speed limit
TURN_SPEED = 10.0    # km/h - speed when turning
BRAKE_SPEED = 0.0    # km/h - speed when braking

# ============================================================================
# PID Controller (matching original InterFuser implementation)
# ============================================================================

class InterFuserPID:
    """
    PID controller matching the ORIGINAL InterFuser implementation exactly.
    Uses a sliding window mean for integral (NOT traditional integral sum).
    See: interfuser_controller.py from opendilab/InterFuser
    """

    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D
        self._window = deque([0 for _ in range(n)], maxlen=n)

    def step(self, error):
        self._window.append(error)
        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = self._window[-1] - self._window[-2]
        else:
            integral = 0.0
            derivative = 0.0
        return self._K_P * error + self._K_I * integral + self._K_D * derivative

    def reset(self):
        self._window.clear()


# ============================================================================
# CARLA Sensor Synchronization
# ============================================================================

class CARLASensorSynchronizer(object):
    """Context manager to synchronize output from different sensors"""

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


# ============================================================================
# InterFuser Model Wrapper
# ============================================================================

class InterfuserModel:
    """Wrapper for InterFuser model loading, preprocessing, and inference"""

    def __init__(self, checkpoint_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Create model
        self.model = create_model('interfuser_baseline', pretrained=False)

        # Load checkpoint
        print(f"Loading checkpoint from: {checkpoint_path}")
        load_checkpoint(self.model, checkpoint_path, strict=True)

        self.model.to(self.device)
        self.model.eval()

        # Setup RGB transform
        self.rgb_transform = create_carla_rgb_transform(
            input_size=224,
            is_training=False,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD
        )

        print("InterFuser model loaded successfully")

    def preprocess_rgb(self, carla_image):
        """Convert CARLA RGB image to model input tensor"""
        array = np.frombuffer(carla_image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (carla_image.height, carla_image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        from PIL import Image
        pil_image = Image.fromarray(array)
        tensor = self.rgb_transform(pil_image)
        return tensor

    def preprocess_lidar(self, carla_lidar):
        """Convert CARLA LiDAR data to model input tensor.

        CRITICAL: Must negate Y-axis to match training data convention.
        CARLA LiDAR: Y+ = right. Training data: Y+ = left (negated).
        See carla_dataset.py line 235: lidar_unprocessed[:, 1] *= -1
        """
        points = np.frombuffer(carla_lidar.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        lidar_data = points[:, :3].copy()

        # Negate Y-axis to match training data convention (carla_dataset.py:235)
        lidar_data[:, 1] *= -1

        lidar_histogram = lidar_to_histogram_features(lidar_data, crop=224)
        tensor = torch.from_numpy(lidar_histogram)
        return tensor

    def create_measurements(self, speed_kmh, command=4):
        """Create measurements tensor

        Command values (CARLA convention):
            1: LEFT
            2: RIGHT
            3: STRAIGHT
            4: LANEFOLLOW (default)
            5: CHANGELANELEFT
            6: CHANGELANERIGHT
        """
        cmd_one_hot = [0, 0, 0, 0, 0, 0]
        cmd_idx = command - 1
        if 0 <= cmd_idx < 6:
            cmd_one_hot[cmd_idx] = 1

        # Speed in m/s (not km/h) - matching training data format
        speed_ms = speed_kmh / 3.6
        measurements = cmd_one_hot + [speed_ms]
        tensor = torch.FloatTensor(measurements).unsqueeze(0)
        return tensor

    def create_target_point(self, ego_transform, target_location):
        """
        Create target point tensor in ego vehicle coordinates.

        CRITICAL: Must match the coordinate transformation used in training data
        (see carla_dataset.py lines 274-286)

        The training data uses:
        - ego_theta: vehicle heading in radians
        - R = [[cos(π/2 + θ), -sin(π/2 + θ)], [sin(π/2 + θ), cos(π/2 + θ)]]
        - local_point = R.T @ (target - ego)
        """
        ego_x = ego_transform.location.x
        ego_y = ego_transform.location.y
        target_x = target_location.x
        target_y = target_location.y

        ego_theta = math.radians(ego_transform.rotation.yaw)

        R = np.array([
            [np.cos(np.pi / 2 + ego_theta), -np.sin(np.pi / 2 + ego_theta)],
            [np.sin(np.pi / 2 + ego_theta), np.cos(np.pi / 2 + ego_theta)]
        ])

        local_command_point = np.array([target_x - ego_x, target_y - ego_y])
        local_command_point = R.T.dot(local_command_point)

        # Handle NaN values
        if np.isnan(local_command_point).any():
            local_command_point = np.array([0.0, 0.0])

        tensor = torch.FloatTensor(local_command_point).unsqueeze(0)
        return tensor

    @torch.no_grad()
    def predict(self, rgb_front, rgb_left, rgb_right, rgb_center, lidar,
                speed_kmh, ego_transform, target_location, command=4):
        """
        Run model inference.

        Args:
            rgb_front: Front camera image
            rgb_left: Left camera image (-60°)
            rgb_right: Right camera image (+60°)
            rgb_center: Center camera image (for narrow FOV)
            lidar: LiDAR point cloud data
            speed_kmh: Current vehicle speed in km/h
            ego_transform: Vehicle transform (position and rotation)
            target_location: Target waypoint from route planner (carla.Location)
            command: Navigation command (1-6, default 4=LANEFOLLOW)

        Returns:
            waypoints: Predicted future waypoints in vehicle coordinates
            is_junction_prob: Probability of being at a junction
            red_light_prob: Probability of red traffic light
            stop_sign_prob: Probability of stop sign
        """
        rgb_tensor = self.preprocess_rgb(rgb_front).unsqueeze(0).to(self.device)
        rgb_left_tensor = self.preprocess_rgb(rgb_left).unsqueeze(0).to(self.device)
        rgb_right_tensor = self.preprocess_rgb(rgb_right).unsqueeze(0).to(self.device)
        rgb_center_tensor = self.preprocess_rgb(rgb_center).unsqueeze(0).to(self.device)
        lidar_tensor = self.preprocess_lidar(lidar).unsqueeze(0).to(self.device)

        measurements = self.create_measurements(speed_kmh, command).to(self.device)
        target_point = self.create_target_point(ego_transform, target_location).to(self.device)

        input_data = {
            'rgb': rgb_tensor,
            'rgb_left': rgb_left_tensor,
            'rgb_right': rgb_right_tensor,
            'rgb_center': rgb_center_tensor,
            'lidar': lidar_tensor,
            'measurements': measurements,
            'target_point': target_point
        }

        traffic, waypoints, is_junction, traffic_light, stop_sign, _ = self.model(input_data)

        waypoints_np = waypoints[0].cpu().numpy()

        # Apply softmax - matching original InterFuser convention:
        #   traffic_light[0] = P(red light), is_junction[0] = P(not junction),
        #   stop_sign[0] = P(not stop sign)
        is_junction_prob = torch.softmax(is_junction[0], dim=-1)[0].item()
        red_light_prob = torch.softmax(traffic_light[0], dim=-1)[0].item()
        stop_sign_prob = torch.softmax(stop_sign[0], dim=-1)[0].item()

        return waypoints_np, is_junction_prob, red_light_prob, stop_sign_prob


# ============================================================================
# Decision Reasoning Engine
# ============================================================================

class DecisionReasoningEngine:
    """Converts model predictions to human-readable decision reasoning"""

    def __init__(self):
        self.prev_reason = "Initializing..."

    def generate_reasoning(self, is_junction_prob, red_light_prob, stop_sign_prob,
                          brake, speed_kmh, waypoints, prev_speed_kmh):
        """Generate human-readable reasoning.

        NOTE on probability conventions (matching original InterFuser):
          red_light_prob = softmax[0] = P(red light present)  → high = stop
          stop_sign_prob = softmax[0] = P(NOT at stop sign)   → low = stop sign present
          is_junction_prob = softmax[0] = P(NOT at junction)  → low = at junction
        """
        deceleration = prev_speed_kmh - speed_kmh
        stop_sign_present = (1.0 - stop_sign_prob)
        at_junction = (1.0 - is_junction_prob)

        if red_light_prob > 0.3:
            reason = "Red traffic light detected"
        elif stop_sign_present > 0.3:
            reason = "Stop sign ahead"
        elif at_junction > 0.5 and speed_kmh < 20:
            reason = "Approaching junction - slowing down"
        elif len(waypoints) > 3:
            lateral_deviation = abs(waypoints[3][1])
            if lateral_deviation > 5.0:
                direction = "left" if waypoints[3][1] > 0 else "right"
                reason = f"Turning {direction}"
            elif brake or deceleration > 0.5:
                reason = "Braking - obstacle detected"
            elif speed_kmh < 5:
                reason = "Stopped - waiting"
            elif speed_kmh > 50:
                reason = "Clear road - cruising"
            else:
                reason = "Normal driving"
        elif brake or deceleration > 0.5:
            reason = "Braking - obstacle detected"
        elif speed_kmh < 5:
            # Calculate approximate distance (placeholder)
            reason = "Stopped (Obstacle 1.9m)"
        elif speed_kmh > 50:
            reason = "Clear road - cruising"
        else:
            reason = "Normal driving"

        self.prev_reason = reason
        return reason


# ============================================================================
# REDESIGNED Panel Renderers
# ============================================================================

class LeftPanelRenderer:
    """
    LEFT PANEL: INTERFUSER PRO Style
    - Title
    - Mode & Speed
    - Controls (sliders)
    - Decision Reasoning
    - LiDAR BEV (Front)
    """

    def __init__(self, width=350, height=720):
        self.width = width
        self.height = height

        # Fonts (compact sizes for better space utilization)
        self.font_title = pygame.font.Font(pygame.font.get_default_font(), 24)
        self.font_large = pygame.font.Font(pygame.font.get_default_font(), 18)
        self.font_medium = pygame.font.Font(pygame.font.get_default_font(), 14)
        self.font_small = pygame.font.Font(pygame.font.get_default_font(), 11)

        # Colors
        self.color_bg = (0, 0, 0)
        self.color_title = (255, 255, 255)
        self.color_green = (0, 255, 0)
        self.color_orange = (255, 165, 0)
        self.color_red = (255, 0, 0)
        self.color_yellow = (255, 255, 0)

    def render(self, surface, speed_kmh, brake, reasoning, lidar_data, total_vehicles=1):
        """Render left panel"""
        surface.fill(self.color_bg)
        y = 8

        # Title
        title = self.font_title.render("INTERFUSER PRO", True, self.color_title)
        surface.blit(title, (10, y))
        y += 32

        # Mode
        mode_text = self.font_medium.render("Mode: AI (Auto)", True, self.color_green)
        surface.blit(mode_text, (10, y))
        y += 22

        # Total Vehicles in Server
        vehicles_text = self.font_medium.render(f"Total Vehicles: {total_vehicles}", True, self.color_yellow)
        surface.blit(vehicles_text, (10, y))
        y += 22

        # Speed
        speed_text = self.font_medium.render(f"Speed: {speed_kmh:.1f} km/h", True, self.color_title)
        surface.blit(speed_text, (10, y))
        y += 28

        # CONTROLS
        controls_title = self.font_large.render("CONTROLS", True, self.color_title)
        surface.blit(controls_title, (10, y))
        y += 24

        # Slider 1 (Green) - Throttle
        throttle_label = self.font_small.render("Throttle", True, self.color_green)
        surface.blit(throttle_label, (10, y + 2))
        pygame.draw.rect(surface, (50, 50, 50), (70, y, 260, 16))
        throttle_width = int(260 * (speed_kmh / 100.0))
        pygame.draw.rect(surface, self.color_green, (70, y, throttle_width, 16))
        y += 22

        # Slider 2 (Orange) - Brake
        brake_label = self.font_small.render("Brake", True, self.color_orange)
        surface.blit(brake_label, (10, y + 2))
        pygame.draw.rect(surface, (50, 50, 50), (70, y, 260, 16))
        if brake:
            pygame.draw.rect(surface, self.color_orange, (70, y, 130, 16))
        y += 28

        # DECISION REASONING
        decision_title = self.font_large.render("DECISION REASONING", True, self.color_title)
        surface.blit(decision_title, (10, y))
        y += 24

        # Reasoning text (red if stopped/obstacle)
        reason_color = self.color_red if "Stopped" in reasoning or "obstacle" in reasoning.lower() else self.color_green
        reason_text = self.font_medium.render(reasoning, True, reason_color)
        surface.blit(reason_text, (10, y))
        y += 30

        # LiDAR BEV (Front)
        lidar_title = self.font_medium.render("LiDAR BEV (Front)", True, self.color_title)
        surface.blit(lidar_title, (10, y))
        y += 22

        # Render LiDAR radar-style view
        self.render_lidar_bev(surface, lidar_data, y)

    def render_lidar_bev(self, surface, lidar_data, y_offset):
        """Render LiDAR Bird's Eye View in radar style (optimized size)"""
        # Parse LiDAR points
        points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        lidar_2d = points[:, :2]

        # Radar circle center and radius (optimized to fill remaining space)
        center_x = self.width // 2
        center_y = y_offset + 170
        max_radius = 165

        # Draw radar circles
        for i in range(1, 5):
            radius = int(max_radius * i / 4)
            pygame.draw.circle(surface, (0, 100, 255), (center_x, center_y), radius, 1)

        # Draw crosshair
        pygame.draw.line(surface, (0, 100, 255), (center_x - max_radius, center_y), (center_x + max_radius, center_y), 1)
        pygame.draw.line(surface, (0, 100, 255), (center_x, center_y - max_radius), (center_x, center_y + max_radius), 1)

        # Plot LiDAR points
        lidar_range = 50.0
        pixels_per_meter = max_radius / lidar_range

        for point in lidar_2d:
            px = int(center_x + point[1] * pixels_per_meter)
            py = int(center_y - point[0] * pixels_per_meter)

            if 0 <= px < self.width and y_offset <= py < self.height:
                distance = np.sqrt(point[0]**2 + point[1]**2)
                if distance < lidar_range:
                    color = (0, 255, 0) if distance < 10 else (0, 200, 0)
                    pygame.draw.circle(surface, color, (px, py), 1)

        # Draw ego vehicle (red circle at center)
        pygame.draw.circle(surface, (255, 0, 0), (center_x, center_y), 5, -1)


class CenterPanelRenderer:
    """
    CENTER PANEL: Main Front Camera View
    Large display with waypoint overlay
    """

    def __init__(self, width=900, height=720):
        self.width = width
        self.height = height

    def render(self, surface, camera_image, waypoints):
        """Render center panel with camera view (fills entire panel with high quality)"""
        # Convert CARLA image to numpy
        array = np.frombuffer(camera_image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (camera_image.height, camera_image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        array = array.copy()

        # Convert to Pygame surface
        img_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        # Scale to FILL panel (crop if necessary to avoid black bars)
        img_width, img_height = img_surface.get_size()
        scale = max(self.width / img_width, self.height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        # Use smoothscale for better quality (slower but much sharper)
        scaled = pygame.transform.smoothscale(img_surface, (new_width, new_height))

        # Center crop the image to fit panel exactly
        x_offset = (self.width - new_width) // 2
        y_offset = (self.height - new_height) // 2

        surface.fill((0, 0, 0))
        surface.blit(scaled, (x_offset, y_offset))

    def draw_waypoints(self, image, waypoints):
        """Draw waypoint overlay on image"""
        img_h, img_w = image.shape[:2]

        for i, wp in enumerate(waypoints):
            # Transform to image coordinates
            pixel_x = int(img_w / 2 - wp[1] * img_w / 30)
            pixel_y = int(img_h - wp[0] * img_h / 30)

            pixel_x = np.clip(pixel_x, 0, img_w - 1)
            pixel_y = np.clip(pixel_y, 0, img_h - 1)

            # Draw with decreasing opacity
            alpha = 1.0 - (i / 10.0)
            color = (0, int(255 * alpha), int(255 * alpha))
            radius = max(3, int(8 * alpha))
            cv2.circle(image, (pixel_x, pixel_y), radius, color, -1)

        return image


class RightPanelRenderer:
    """
    RIGHT PANEL: 3 Small Camera Views
    FRONT, LEFT, RIGHT - stacked vertically (REAR removed, it's in center panel)
    """

    def __init__(self, width=350, height=720):
        self.width = width
        self.height = height
        self.camera_height = height // 3  # Changed to 3 cameras
        self.font = pygame.font.Font(pygame.font.get_default_font(), 14)

    def render(self, surface, front_img, left_img, right_img):
        """Render right panel with three camera views"""
        surface.fill((0, 0, 0))

        # Render 3 cameras (REAR removed)
        self._render_camera(surface, front_img, 0, "FRONT")
        self._render_camera(surface, left_img, self.camera_height, "LEFT")
        self._render_camera(surface, right_img, self.camera_height * 2, "RIGHT")

    def _render_camera(self, surface, camera_image, y_offset, label):
        """Render single camera view filling the entire space"""
        # Convert CARLA image to numpy
        array = np.frombuffer(camera_image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (camera_image.height, camera_image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        # Convert to Pygame surface
        img_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        # Scale to FILL the space (crop if necessary)
        img_width, img_height = img_surface.get_size()
        scale = max(self.width / img_width, self.camera_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        scaled = pygame.transform.scale(img_surface, (new_width, new_height))

        # Center crop
        x_offset = (self.width - new_width) // 2
        y_center = y_offset + (self.camera_height - new_height) // 2

        # Draw black background
        pygame.draw.rect(surface, (0, 0, 0), (0, y_offset, self.width, self.camera_height))

        # Blit image (may overflow, but will be clipped)
        surface.blit(scaled, (x_offset, y_center))

        # Draw label with yellow background
        label_surf = self.font.render(label, True, (0, 0, 0))
        label_bg = pygame.Surface((self.width, 20))
        label_bg.fill((255, 255, 0))
        label_bg.set_alpha(200)
        surface.blit(label_bg, (0, y_offset))
        surface.blit(label_surf, (5, y_offset + 2))


# ============================================================================
# Main UI Manager
# ============================================================================

class RoutePlanner:
    """
    Route planner using CARLA's GlobalRoutePlanner.
    Plans a global route to a destination and follows it with proper commands.
    """

    def __init__(self, vehicle, world):
        from agents.navigation.global_route_planner import GlobalRoutePlanner

        self._vehicle = vehicle
        self._world = world
        self._map = world.get_map()
        self._grp = GlobalRoutePlanner(self._map, sampling_resolution=2.0)
        self._route = deque()
        self._min_distance = 5.0

        self._plan_new_route()

    def _plan_new_route(self):
        """Plan a route to a random destination on the map."""
        spawn_points = self._map.get_spawn_points()
        if not spawn_points:
            return

        start_location = self._vehicle.get_transform().location
        destination = random.choice(spawn_points).location

        # Ensure destination is far enough away
        for _ in range(10):
            dx = destination.x - start_location.x
            dy = destination.y - start_location.y
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > 50.0:
                break
            destination = random.choice(spawn_points).location

        try:
            route = self._grp.trace_route(start_location, destination)
            self._route = deque(route)
            print(f"Route planned: {len(self._route)} waypoints, dist ~{dist:.0f}m")
        except Exception as e:
            print(f"Route planning failed: {e}")
            self._route = deque()

    def run_step(self):
        """
        Get next waypoint and command. Pops reached waypoints.
        Returns (carla.Location, command_int).
        Command values: 1=LEFT, 2=RIGHT, 3=STRAIGHT, 4=LANEFOLLOW
        """
        vehicle_location = self._vehicle.get_transform().location

        # Pop waypoints we've already passed
        while len(self._route) > 2:
            wp, _ = self._route[0]
            wp_loc = wp.transform.location
            dx = wp_loc.x - vehicle_location.x
            dy = wp_loc.y - vehicle_location.y
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < self._min_distance:
                self._route.popleft()
            else:
                break

        # If route is nearly exhausted, plan a new one
        if len(self._route) < 5:
            self._plan_new_route()

        # Return the next waypoint on the route
        if len(self._route) > 1:
            wp, road_option = self._route[1]
            cmd = int(road_option.value)
            # Clamp command to valid range 1-6
            if cmd < 1 or cmd > 6:
                cmd = 4  # default LANEFOLLOW
            return wp.transform.location, cmd

        # Fallback: use current road waypoint
        current_wp = self._map.get_waypoint(vehicle_location)
        if current_wp:
            next_wps = current_wp.next(20.0)
            if next_wps:
                return next_wps[0].transform.location, 4
        return vehicle_location, 4


class InterfuserUIManager:
    """Main orchestrator for InterFuser real-time monitoring UI"""

    def __init__(self, host='localhost', port=2000,
                 checkpoint_path=os.path.join(PROJECT_ROOT, 'checkpoints', 'interfuser.pth')):
        # Initialize Pygame
        pygame.init()
        self.display = pygame.display.set_mode(
            (1600, 720),
            pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE
        )
        pygame.display.set_caption("InterFuser Real-time Monitoring")

        # Initialize panels with wider dimensions
        self.left_panel = LeftPanelRenderer(width=350, height=720)
        self.center_panel = CenterPanelRenderer(width=900, height=720)
        self.right_panel = RightPanelRenderer(width=350, height=720)

        # CARLA setup
        print("Connecting to CARLA server...")
        self.client = carla.Client(host, port)
        self.client.set_timeout(30.0)  # Increased timeout for slower connections
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        print(f"Connected to CARLA: {self.map.name}")

        # Spawn vehicle and sensors
        self.ego_vehicle = None
        self.sensors = []
        self.spawn_vehicle()
        self.setup_sensors()

        # Initialize route planner for proper navigation
        self.route_planner = RoutePlanner(self.ego_vehicle, self.world)

        # Initialize model
        print("Loading InterFuser model...")
        self.model = InterfuserModel(checkpoint_path)

        # Initialize reasoning
        self.reasoning_engine = DecisionReasoningEngine()

        # Initialize PID controllers (matching original InterFuser exactly)
        self.turn_controller = InterFuserPID(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self.speed_pid = InterFuserPID(K_P=5.0, K_I=0.5, K_D=1.0, n=40)

        # State tracking
        self.clock = pygame.time.Clock()
        self.prev_speed = 0

        # Safety: startup grace period (hold brake until model stabilizes)
        self.STARTUP_GRACE_FRAMES = 20      # Hold brake for first N frames
        self.startup_frame = 0

        # Stop sign / traffic light state machine
        self.stop_sign_stopped_time = None  # When we first stopped for stop sign
        self.stop_sign_cooldown_until = 0   # Ignore stop signs until this time
        self.STOP_SIGN_WAIT = 3.0           # Seconds to wait at stop sign
        self.STOP_SIGN_COOLDOWN = 15.0      # Seconds to ignore stop signs after resuming

        print("Initialization complete!")

    def spawn_vehicle(self):
        """Spawn ego vehicle"""
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.*')[0]

        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = spawn_points[0] if spawn_points else carla.Transform()

        self.ego_vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        # DO NOT use autopilot - InterFuser will control the vehicle
        print(f"Spawned vehicle: {vehicle_bp.id}")

    def setup_sensors(self):
        """Setup all required sensors"""
        blueprint_library = self.world.get_blueprint_library()

        # 1. Rear Camera (Ultra high resolution for center panel - 2K)
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '2560')
        camera_bp.set_attribute('image_size_y', '1440')
        camera_bp.set_attribute('fov', '90')
        rear_cam = self.world.spawn_actor(
            camera_bp,
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=self.ego_vehicle
        )
        self.sensors.append(rear_cam)

        # 2. Left Camera (-60°) - Higher resolution for right panel
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1280')
        camera_bp.set_attribute('image_size_y', '720')
        camera_bp.set_attribute('fov', '90')
        left_cam = self.world.spawn_actor(
            camera_bp,
            carla.Transform(carla.Location(x=1.6, z=1.6), carla.Rotation(yaw=-60)),
            attach_to=self.ego_vehicle
        )
        self.sensors.append(left_cam)

        # 3. Front Camera (0°) - Higher resolution for right panel
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1280')
        camera_bp.set_attribute('image_size_y', '720')
        camera_bp.set_attribute('fov', '90')
        front_cam = self.world.spawn_actor(
            camera_bp,
            carla.Transform(carla.Location(x=1.6, z=1.6), carla.Rotation(yaw=0)),
            attach_to=self.ego_vehicle
        )
        self.sensors.append(front_cam)

        # 4. Right Camera (+60°) - Higher resolution for right panel
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1280')
        camera_bp.set_attribute('image_size_y', '720')
        camera_bp.set_attribute('fov', '90')
        right_cam = self.world.spawn_actor(
            camera_bp,
            carla.Transform(carla.Location(x=1.6, z=1.6), carla.Rotation(yaw=60)),
            attach_to=self.ego_vehicle
        )
        self.sensors.append(right_cam)

        # 5. Front-Center Camera (for InterFuser input)
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '341')
        camera_bp.set_attribute('image_size_y', '256')
        camera_bp.set_attribute('fov', '90')
        front_center_cam = self.world.spawn_actor(
            camera_bp,
            carla.Transform(carla.Location(x=1.6, z=1.6), carla.Rotation(yaw=0)),
            attach_to=self.ego_vehicle
        )
        self.sensors.append(front_center_cam)

        # 6. LiDAR
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '100')
        lidar_bp.set_attribute('rotation_frequency', '20')
        lidar_bp.set_attribute('channels', '64')
        lidar_bp.set_attribute('points_per_second', '100000')
        lidar_bp.set_attribute('upper_fov', '10')
        lidar_bp.set_attribute('lower_fov', '-30')
        lidar = self.world.spawn_actor(
            lidar_bp,
            carla.Transform(carla.Location(x=1.0, z=1.8)),
            attach_to=self.ego_vehicle
        )
        self.sensors.append(lidar)

        print(f"Spawned {len(self.sensors)} sensors")

    def should_quit(self):
        """Check if user wants to quit"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_ESCAPE:
                    return True
        return False

    def run(self):
        """Main loop"""
        try:
            with CARLASensorSynchronizer(self.world, *self.sensors, fps=20) as sync:
                print("Starting main loop...")
                frame_count = 0

                while True:
                    if self.should_quit():
                        break

                    # Tick simulation and get sensor data
                    # Order: rear_cam, left_cam, front_cam, right_cam, front_center_cam, lidar
                    snapshot, rear_cam, left_cam, front_cam, right_cam, front_center_cam, lidar = sync.tick(timeout=2.0)

                    # Get vehicle state
                    velocity = self.ego_vehicle.get_velocity()
                    speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

                    # Get target waypoint and command from route planner
                    ego_transform = self.ego_vehicle.get_transform()
                    target_location, command = self.route_planner.run_step()

                    # Run model inference with proper target_point and command
                    waypoints, is_junction_prob, red_light_prob, stop_sign_prob = self.model.predict(
                        front_cam, left_cam, right_cam, front_center_cam, lidar,
                        speed, ego_transform, target_location, command
                    )

                    # Calculate control from InterFuser predictions and apply to vehicle
                    control = self._calculate_control(
                        waypoints, red_light_prob, stop_sign_prob, speed
                    )
                    self.ego_vehicle.apply_control(control)

                    # Get brake status from control
                    brake = control.brake > 0.1

                    # Generate reasoning
                    reasoning = self.reasoning_engine.generate_reasoning(
                        is_junction_prob, red_light_prob, stop_sign_prob,
                        brake, speed, waypoints, self.prev_speed
                    )

                    # Render UI
                    self.render_ui(
                        speed, brake, reasoning,
                        front_cam, left_cam, right_cam, rear_cam, lidar, waypoints
                    )

                    # Update display
                    pygame.display.flip()
                    self.clock.tick(20)

                    # Update state
                    self.prev_speed = speed
                    frame_count += 1

                    # Print FPS only (debug data goes to CSV)
                    if frame_count % 100 == 0:
                        print(f"Frame {frame_count} | FPS: {self.clock.get_fps():.1f} | Speed: {speed:.1f} km/h")

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.cleanup()

    def _calculate_control(self, waypoints, red_light_prob, stop_sign_prob,
                           current_speed):
        """Calculate vehicle control from InterFuser model outputs."""
        control = carla.VehicleControl()
        now = time.time()

        angle = 0.0
        steer = 0.0
        throttle = 0.0
        brake_val = 0.0
        target_speed = TARGET_SPEED

        # =====================================================================
        # 1. STEERING (exact match to original InterFuser)
        # =====================================================================
        if len(waypoints) >= 2:
            aim = (waypoints[1] + waypoints[0]) / 2.0
            aim[1] *= -1

            angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90.0

            if current_speed < 0.04:
                angle = 0.0

            steer = self.turn_controller.step(angle)
            steer = np.clip(steer, -1.0, 1.0)
            control.steer = float(steer)
        else:
            control.steer = 0.0

        # =====================================================================
        # 2. BRAKE DECISION with safety rules
        # =====================================================================
        should_brake = False
        stop_sign_detected = (1.0 - stop_sign_prob) > 0.3
        ss_cooldown_active = now < self.stop_sign_cooldown_until

        # --- Safety Rule 0: Startup grace period ---
        self.startup_frame += 1
        if self.startup_frame <= self.STARTUP_GRACE_FRAMES:
            should_brake = True

        # --- Safety Rule 1: CARLA ground-truth red light check ---
        if not should_brake and self.ego_vehicle.is_at_traffic_light():
            traffic_light = self.ego_vehicle.get_traffic_light()
            if traffic_light is not None:
                tl_state = traffic_light.get_state()
                if tl_state in (carla.TrafficLightState.Red, carla.TrafficLightState.Yellow):
                    should_brake = True

        # --- Safety Rule 2: Model-based red light ---
        if not should_brake and red_light_prob > 0.25:
            should_brake = True

        # --- Safety Rule 3: Stop sign with timed resume ---
        if not should_brake and stop_sign_detected and not ss_cooldown_active:
            if current_speed < 0.5:
                if self.stop_sign_stopped_time is None:
                    self.stop_sign_stopped_time = now
                elapsed = now - self.stop_sign_stopped_time
                if elapsed < self.STOP_SIGN_WAIT:
                    should_brake = True
                else:
                    self.stop_sign_cooldown_until = now + self.STOP_SIGN_COOLDOWN
                    self.stop_sign_stopped_time = None
            else:
                should_brake = True
                self.stop_sign_stopped_time = None
        elif not stop_sign_detected or ss_cooldown_active:
            self.stop_sign_stopped_time = None

        if should_brake:
            target_speed = 0.0
        else:
            target_speed = TARGET_SPEED

        # =====================================================================
        # 3. THROTTLE / BRAKE (PID speed control + max speed limit)
        # =====================================================================
        if not should_brake and current_speed > MAX_SPEED:
            should_brake = True
            target_speed = 0.0

        if should_brake:
            throttle = 0.0
            brake_val = 1.0
        else:
            # Cap target speed to MAX_SPEED
            effective_target = min(target_speed, MAX_SPEED)
            speed_error = effective_target - current_speed
            if speed_error > 0:
                throttle = np.clip(self.speed_pid.step(speed_error), 0.0, 0.8)
                brake_val = 0.0
            else:
                throttle = 0.0
                brake_val = np.clip(-self.speed_pid.step(speed_error), 0.0, 0.8)

        control.throttle = float(throttle)
        control.brake = float(brake_val)
        control.hand_brake = False
        control.manual_gear_shift = False

        return control

    def render_ui(self, speed, brake, reasoning,
                  front_cam, left_cam, right_cam, rear_cam, lidar, waypoints):
        """Render all three panels"""
        # Clear display
        self.display.fill((0, 0, 0))

        # Get total number of vehicles in the server
        vehicles = self.world.get_actors().filter('vehicle.*')
        total_vehicles = len(vehicles)

        # Create panel surfaces with wider dimensions
        left_surface = pygame.Surface((350, 720))
        center_surface = pygame.Surface((900, 720))
        right_surface = pygame.Surface((350, 720))

        # Render panels
        self.left_panel.render(left_surface, speed, brake, reasoning, lidar, total_vehicles)
        self.center_panel.render(center_surface, rear_cam, waypoints)  # Changed to rear_cam per user request
        self.right_panel.render(right_surface, front_cam, left_cam, right_cam)  # Removed rear_cam

        # Blit to main display with new positions
        self.display.blit(left_surface, (0, 0))
        self.display.blit(center_surface, (350, 0))
        self.display.blit(right_surface, (1250, 0))

    def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up...")

        for sensor in self.sensors:
            if sensor.is_alive:
                sensor.destroy()

        if self.ego_vehicle and self.ego_vehicle.is_alive:
            self.ego_vehicle.destroy()

        pygame.quit()
        print("Cleanup complete")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point"""
    try:
        ui_manager = InterfuserUIManager()
        ui_manager.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()