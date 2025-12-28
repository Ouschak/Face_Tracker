import logging
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options


class LookingSensor:
    def __init__(
        self,
        model_path="models/face_landmarker.task",
        debug=True,
        padding_pct=0.10,
        look_threshold=0.20,  # threshold for looking forward
        smooth_alpha=0.7,  
    ):
        self.log = logging.getLogger(self.__class__.__name__)
        self.debug = debug
        self.padding_pct = float(padding_pct)

        self.look_threshold = float(look_threshold)
        self.smooth_alpha = float(smooth_alpha)
        self.moving_avg = None 

        options = vision.FaceLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)

        self._t0 = time.monotonic()
        self._last_ts_ms = -1

    def process_frame(self, frame_bgr):
        if frame_bgr is None:
            return {"face_present": False}

        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        timestamp_ms = int((time.monotonic() - self._t0) * 1000)
        if timestamp_ms <= self._last_ts_ms:
            timestamp_ms = self._last_ts_ms + 1
        self._last_ts_ms = timestamp_ms

        result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            return {"face_present": False}

        landmarks = result.face_landmarks[0]

        leftmost = min(landmarks, key=lambda lm: lm.x)
        rightmost = max(landmarks, key=lambda lm: lm.x)
        topmost = min(landmarks, key=lambda lm: lm.y)
        bottommost = max(landmarks, key=lambda lm: lm.y)

        face_width_norm = abs(rightmost.x - leftmost.x)
        face_height_norm = abs(bottommost.y - topmost.y)

        nose = landmarks[1]

        center_x = (leftmost.x + rightmost.x) / 2.0
        offset_norm = (nose.x - center_x) / face_width_norm

        # Smooth it (EMA)
        if self.moving_avg is None:
            self.moving_avg = offset_norm
        else:
            a = self.smooth_alpha
            self.moving_avg = a * self.moving_avg + (1.0 - a) * offset_norm

        looking = abs(self.moving_avg) < self.look_threshold

        pad_x = self.padding_pct * face_width_norm
        pad_y = self.padding_pct * face_height_norm
        x1 = int(max(0, (leftmost.x - pad_x) * w))
        x2 = int(min(w - 1, (rightmost.x + pad_x) * w))
        y1 = int(max(0, (topmost.y - pad_y) * h))
        y2 = int(min(h - 1, (bottommost.y + pad_y) * h))

        if self.debug:
            for lm in landmarks:
                cv2.circle(
                    frame_bgr, (int(lm.x * w), int(lm.y * h)), 1, (0, 255, 0), -1
                )

            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 255, 0), 1)

            # Mark nose and center
            cv2.circle(
                frame_bgr, (int(nose.x * w), int(nose.y * h)), 4, (0, 0, 255), -1
            )
            cv2.circle(
                frame_bgr, (int(center_x * w), int(nose.y * h)), 4, (255, 0, 0), -1
            )

            cv2.putText(
                frame_bgr,
                f"looking={looking} offset={self.moving_avg:.3f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        return {
            "face_present": True,
            "looking": looking,
        }

    def close(self):
        self.log.info("Closing FaceLandmarker")
        self.face_landmarker.close()
