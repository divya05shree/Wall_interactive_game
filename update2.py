import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import mediapipe as mp
import socket
import json
import time


class ProjectorHandDistanceDetector:
    def __init__(self, model_path='best.pt'):
        self.model = YOLO(model_path)
        self.projector_box = None
        self.projector_distance = None

        # RealSense setup
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        # UDP
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_ip = "127.0.0.1"
        self.port = 5053

        # MediaPipe Hands (single hand)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Stability filters
        self.prev_mapped = None
        self.last_depth = None
        self.stable_count = 0
        self.required_stable_frames = 1  # lower for responsiveness
        self.same_frame_counter = 0
        self.max_static_frames = 30  # if exactly same for too long, treat as ghost

    def get_aligned_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None, None
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_colormap = cv2.convertScaleAbs(depth_image, alpha=0.03)
        return color_image, depth_frame, depth_colormap

    def detect_projector_once(self, image, depth_frame):
        if self.projector_box is not None:
            return self.projector_box

        try:
            results = self.model.predict(image, imgsz=640, conf=0.5, verbose=False)
        except Exception as e:
            print("⚠️ YOLO inference failed:", e)
            return None

        boxes = results[0].boxes
        max_area = 0
        img_h, img_w, _ = image.shape
        for box in boxes:
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            if label == "bright_screen":
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Clamp to image bounds
                x1 = max(0, min(x1, img_w - 1))
                x2 = max(0, min(x2, img_w - 1))
                y1 = max(0, min(y1, img_h - 1))
                y2 = max(0, min(y2, img_h - 1))
                if x2 <= x1 or y2 <= y1:
                    continue

                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    self.projector_box = (x1, y1, x2, y2)
                    max_area = area

        if self.projector_box:
            cx = (self.projector_box[0] + self.projector_box[2]) // 2
            cy = (self.projector_box[1] + self.projector_box[3]) // 2
            try:
                self.projector_distance = depth_frame.get_distance(cx, cy)
            except Exception:
                self.projector_distance = None

        return self.projector_box

    def detect_palm_center_and_depth(self, image, depth_frame):
        start_t = time.time()
        if self.projector_box is None:
            self._reset_stability()
            return None, None, None

        x1, y1, x2, y2 = self.projector_box
        img_h, img_w, _ = image.shape

        # Clamp projector box
        x1c = max(0, min(x1, img_w - 1))
        x2c = max(0, min(x2, img_w - 1))
        y1c = max(0, min(y1, img_h - 1))
        y2c = max(0, min(y2, img_h - 1))
        if x2c <= x1c or y2c <= y1c:
            self._reset_stability()
            return None, None, None

        roi = image[y1c:y2c, x1c:x2c]
        if roi.size == 0:
            self._reset_stability()
            return None, None, None

        # Downscale ROI for faster MediaPipe inference
        small_roi = cv2.resize(roi, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        rgb_small = cv2.cvtColor(small_roi, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_small)

        if not results.multi_hand_landmarks or not results.multi_handedness:
            self._reset_stability()
            return None, None, None

        hand_landmarks = results.multi_hand_landmarks[0]
        handedness = results.multi_handedness[0]
        score = handedness.classification[0].score
        if score < 0.8:  # slightly relaxed for responsiveness
            self._reset_stability()
            return None, None, None

        # Normalized landmark bounding box for noise rejection
        lm_x = [lm.x for lm in hand_landmarks.landmark]
        lm_y = [lm.y for lm in hand_landmarks.landmark]
        min_x, max_x = min(lm_x), max(lm_x)
        min_y, max_y = min(lm_y), max(lm_y)
        box_w_norm = max_x - min_x
        box_h_norm = max_y - min_y
        if box_w_norm < 0.03 or box_h_norm < 0.03 or box_w_norm > 0.9 or box_h_norm > 0.9:
            self._reset_stability()
            return None, None, None

        # Palm center (on small ROI)
        palm_indices = [0, 5, 9, 13, 17]
        small_h, small_w, _ = small_roi.shape
        cx_small = int(np.mean([hand_landmarks.landmark[i].x for i in palm_indices]) * small_w)
        cy_small = int(np.mean([hand_landmarks.landmark[i].y for i in palm_indices]) * small_h)

        # Scale back to original ROI coordinates
        full_x = x1c + cx_small * 2
        full_y = y1c + cy_small * 2

        try:
            depth_m = depth_frame.get_distance(full_x, full_y)
        except Exception:
            self._reset_stability()
            return None, None, None

        # Depth plausibility check
        if self.projector_distance is None:
            self._reset_stability()
            return None, None, None
        diff = abs(self.projector_distance - depth_m)
        if diff > 1.0:  # tune as needed
            self._reset_stability()
            return None, None, None

        # Map to projector output resolution (1920x1080)
        box_width = x2c - x1c
        box_height = y2c - y1c
        if box_width == 0 or box_height == 0:
            self._reset_stability()
            return None, None, None

        mapped_x = map_value(0, box_width, 0, 1920, full_x - x1c)
        mapped_y = map_value(0, box_height, 0, 1080, full_y - y1c)
        mapped_pos = [
            int(max(0, min(1919, mapped_x))),
            int(max(0, min(1079, mapped_y)))
        ]

        # Stability logic
        if self.prev_mapped is not None:
            dx = abs(mapped_pos[0] - self.prev_mapped[0])
            dy = abs(mapped_pos[1] - self.prev_mapped[1])
            dd = abs(depth_m - (self.last_depth if self.last_depth is not None else depth_m))
            if dx < 15 and dy < 15 and dd < 0.15:
                self.stable_count += 1
            else:
                self.stable_count = 1
                self.same_frame_counter = 0
        else:
            self.stable_count = 1

        if self.prev_mapped == mapped_pos:
            self.same_frame_counter += 1
        else:
            self.same_frame_counter = 0

        self.prev_mapped = mapped_pos
        self.last_depth = depth_m

        # Accept only if stable enough and not stuck ghost
        if self.stable_count < self.required_stable_frames or self.same_frame_counter > self.max_static_frames:
            return None, None, None

        # Visualization
        cv2.circle(image, (full_x, full_y), 6, (0, 0, 255), -1)
        cv2.putText(
            image,
            f"Palm: {mapped_pos} {depth_m:.3f}m",
            (full_x + 10, full_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        # Logging latency (optional)
        # print(f"[hand detection] {(time.time() - start_t)*1000:.1f}ms")

        return (full_x, full_y), depth_m, mapped_pos

    def _reset_stability(self):
        self.stable_count = 0
        self.same_frame_counter = 0
        self.prev_mapped = None
        self.last_depth = None

    def draw_projector_info(self, image):
        if self.projector_box:
            x1, y1, x2, y2 = self.projector_box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(image, "Projector", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            if self.projector_distance is not None:
                cv2.circle(image, (cx, cy), 6, (255, 0, 255), -1)
                cv2.putText(image, f"Proj: {self.projector_distance:.3f} m", (cx + 10, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 100, 255), 2)

    def stop(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()


def map_value(old_min, old_max, new_min, new_max, value):
    if old_max - old_min == 0:
        return new_min
    old_range = old_max - old_min
    new_range = new_max - new_min
    scaled_value = (value - old_min) / old_range
    return new_min + (scaled_value * new_range)


def main():
    detector = ProjectorHandDistanceDetector('best.pt')

    try:
        while True:
            color_image, depth_frame, depth_colormap = detector.get_aligned_frames()
            if color_image is None:
                continue

            detector.detect_projector_once(color_image, depth_frame)
            detector.draw_projector_info(color_image)
            cy, depth, mapped_pos = detector.detect_palm_center_and_depth(color_image, depth_frame)

            cv2.imshow("Color Image", color_image)
            cv2.imshow("Depth Colormap", depth_colormap)

            if cy is not None and depth is not None and mapped_pos is not None:
                data = {
                    "hand_depth": depth,
                    "projector_depth": detector.projector_distance,
                    "hand_position": [mapped_pos[0], mapped_pos[1]],
                    "ts": time.time()
                }
                message = json.dumps(data).encode("utf-8")
                detector.sock.sendto(message, (detector.server_ip, detector.port))
                print("✅ Sent:", data)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        detector.stop()


if __name__ == "__main__":
    main()
