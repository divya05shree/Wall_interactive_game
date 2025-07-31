import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import mediapipe as mp
import socket
import json


class ProjectorHandDistanceDetector:
    def __init__(self, model_path='best.pt'):
        self.model = YOLO(model_path)
        self.projector_box = None
        self.projector_distance = None

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_ip = "127.0.0.1"
        self.port = 5053

        self.align = rs.align(rs.stream.color)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

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

        results = self.model.predict(image, imgsz=640, conf=0.5, verbose=False)
        boxes = results[0].boxes
        max_area = 0
        for box in boxes:
            cls_id = int(box.cls[0])
            label = self.model.names[cls_id]
            if label == "bright_screen":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    self.projector_box = (x1, y1, x2, y2)
                    max_area = area

        if self.projector_box:
            cx = (self.projector_box[0] + self.projector_box[2]) // 2
            cy = (self.projector_box[1] + self.projector_box[3]) // 2
            self.projector_distance = depth_frame.get_distance(cx, cy)

        return self.projector_box

    def detect_palm_center_and_depth(self, image, depth_frame):
        if self.projector_box is None:
            return None, None, None

        x1, y1, x2, y2 = self.projector_box
        roi = image[y1:y2, x1:x2]
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_roi)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                palm_indices = [0, 5, 9, 13, 17]
                h, w, _ = roi.shape
                cx = int(np.mean([hand_landmarks.landmark[i].x for i in palm_indices]) * w)
                cy = int(np.mean([hand_landmarks.landmark[i].y for i in palm_indices]) * h)

                full_x = x1 + cx
                full_y = y1 + cy

                depth_m = depth_frame.get_distance(full_x, full_y)

                cv2.circle(image, (full_x, full_y), 6, (0, 0, 255), -1)
                cv2.putText(image, f"Palm: {full_x - x1, full_y - y1} {depth_m:.3f} m", (full_x + 10, full_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                box_width = x2 - x1
                box_height = y2 - y1

                if box_width == 0 or box_height == 0:
                    return None, None, None

                mapped_x = map_value(0, box_width, 0, 1920, full_x - x1)
                mapped_y = map_value(0, box_height, 0, 1080, full_y - y1)

                return (full_x, full_y), depth_m, [mapped_x, mapped_y]

        return None, None, None

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

            if cy is not None:
                data = {
                    "hand_depth": depth,
                    "projector_depth": detector.projector_distance,
                    "hand_position": [mapped_pos[0], mapped_pos[1]],
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
