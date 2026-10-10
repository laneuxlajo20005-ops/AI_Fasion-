"""Optional MediaPipe Pose-based styling estimates from transient user images."""

from __future__ import annotations

import base64
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MIN_LANDMARK_VISIBILITY = 0.55
MIN_RATIO_FOR_ESTIMATION = 0.45
MAX_RATIO_FOR_ESTIMATION = 1.9
BODY_SHAPE_LABELS = (
    "Pear/triangle-style",
    "Inverted-triangle-style",
    "Balanced shoulder/hip-style",
)
BODY_SHAPE_CATEGORY_PROXIES = {
    "Pear/triangle-style": frozenset({"Tops", "Outerwear", "Accessories"}),
    "Inverted-triangle-style": frozenset({"Bottoms", "Dresses", "Accessories"}),
    "Balanced shoulder/hip-style": frozenset(
        {"Dresses", "Tops", "Bottoms", "Accessories"}
    ),
}


class BodyShapeError(ValueError):
    """User-facing problem with the supplied image or pose."""


class BodyShapeUnavailable(RuntimeError):
    """Optional computer-vision packages are not installed."""


@dataclass(frozen=True)
class BodyShapeEstimate:
    category: str
    measurements: dict[str, float | None]
    explanation: str
    limitation: str
    landmark_image_data_uri: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _landmark_xy(landmarks: Any, index: int, min_visibility: float) -> tuple[float, float]:
    point = landmarks[index]
    visibility = float(getattr(point, "visibility", 1.0))
    x = float(point.x)
    y = float(point.y)
    if (
        not math.isfinite(x)
        or not math.isfinite(y)
        or not math.isfinite(visibility)
        or visibility < min_visibility
    ):
        raise BodyShapeError(
            "Pose landmarks are unclear. Try a brighter, front-facing full-body image."
        )
    return x, y


def _horizontal_silhouette_width(
    segmentation_mask: np.ndarray,
    normalized_y: float,
    center_x: float,
) -> float | None:
    height, width = segmentation_mask.shape[:2]
    row_index = int(round(normalized_y * (height - 1)))
    if not 0 <= row_index < height:
        return None
    start = max(0, row_index - max(1, round(height * 0.005)))
    end = min(height, row_index + max(2, round(height * 0.005)) + 1)
    mask = segmentation_mask[start:end, :] >= 0.5
    occupied_columns = np.flatnonzero(mask.mean(axis=0) >= 0.5)
    if not len(occupied_columns):
        return None

    groups: list[list[int]] = [[int(occupied_columns[0])]]
    for column in occupied_columns[1:]:
        column = int(column)
        if column <= groups[-1][-1] + 1:
            groups[-1].append(column)
        else:
            groups.append([column])
    center_pixel = center_x * (width - 1)
    body_groups = [
        group for group in groups
        if group[0] <= center_pixel <= group[-1]
    ]
    if not body_groups:
        body_groups = sorted(
            groups,
            key=lambda group: (
                abs((group[0] + group[-1]) / 2 - center_pixel),
                -len(group),
            ),
        )[:1]
    widest = max(body_groups, key=len)
    return float(len(widest) / width)


def measure_pose(
    landmarks: Any,
    image_width: int,
    image_height: int,
    segmentation_mask: np.ndarray | None,
) -> dict[str, float | None]:
    """Extract normalized 2D proportions from visible MediaPipe pose landmarks."""
    if image_width < 1 or image_height < 1:
        raise BodyShapeError("Image dimensions are invalid")
    left_shoulder = _landmark_xy(landmarks, 11, MIN_LANDMARK_VISIBILITY)
    right_shoulder = _landmark_xy(landmarks, 12, MIN_LANDMARK_VISIBILITY)
    left_hip = _landmark_xy(landmarks, 23, MIN_LANDMARK_VISIBILITY)
    right_hip = _landmark_xy(landmarks, 24, MIN_LANDMARK_VISIBILITY)
    left_ankle = _landmark_xy(landmarks, 27, MIN_LANDMARK_VISIBILITY)
    right_ankle = _landmark_xy(landmarks, 28, MIN_LANDMARK_VISIBILITY)

    shoulder_width_px = abs(right_shoulder[0] - left_shoulder[0]) * image_width
    hip_width_px = abs(right_hip[0] - left_hip[0]) * image_width
    shoulder_center = (
        (left_shoulder[0] + right_shoulder[0]) / 2,
        (left_shoulder[1] + right_shoulder[1]) / 2,
    )
    hip_center = (
        (left_hip[0] + right_hip[0]) / 2,
        (left_hip[1] + right_hip[1]) / 2,
    )
    ankle_center = (
        (left_ankle[0] + right_ankle[0]) / 2,
        (left_ankle[1] + right_ankle[1]) / 2,
    )
    torso_length_px = math.hypot(
        (hip_center[0] - shoulder_center[0]) * image_width,
        (hip_center[1] - shoulder_center[1]) * image_height,
    )
    leg_length_px = math.hypot(
        (ankle_center[0] - hip_center[0]) * image_width,
        (ankle_center[1] - hip_center[1]) * image_height,
    )
    if (
        shoulder_width_px <= 0
        or hip_width_px <= 0
        or torso_length_px <= 0
        or leg_length_px <= 0
    ):
        raise BodyShapeError(
            "A clear upright full-body pose was not detected. Include shoulders, hips, and ankles."
        )

    shoulder_hip_ratio = shoulder_width_px / hip_width_px
    if not MIN_RATIO_FOR_ESTIMATION <= shoulder_hip_ratio <= MAX_RATIO_FOR_ESTIMATION:
        raise BodyShapeError(
            "The shoulder-to-hip proportion is unclear. Try a straight-on, uncropped image."
        )
    body_length = torso_length_px + leg_length_px

    waist_proxy: float | None = None
    if segmentation_mask is not None and segmentation_mask.ndim == 2:
        waist_y = shoulder_center[1] + 0.55 * (
            hip_center[1] - shoulder_center[1]
        )
        waist_proxy = _horizontal_silhouette_width(
            segmentation_mask,
            waist_y,
            (shoulder_center[0] + hip_center[0]) / 2,
        )

    return {
        "shoulder_width_over_torso_length": shoulder_width_px / torso_length_px,
        "hip_width_over_torso_length": hip_width_px / torso_length_px,
        "shoulder_to_hip_width_ratio": shoulder_hip_ratio,
        "torso_proportion_of_shoulder_to_ankle": torso_length_px / body_length,
        "leg_proportion_of_shoulder_to_ankle": leg_length_px / body_length,
        "waist_silhouette_proxy_over_hip_width": (
            waist_proxy / (hip_width_px / image_width)
            if waist_proxy is not None
            else None
        ),
    }


def classify_styling_shape(measurements: dict[str, float | None]) -> tuple[str, str]:
    """Apply a cautious shoulder/hip-only shape proxy, not an objective body type."""
    ratio = measurements["shoulder_to_hip_width_ratio"]
    if ratio is None or not math.isfinite(float(ratio)):
        raise BodyShapeError("Shoulder-to-hip proportion could not be estimated")
    if ratio < 0.90:
        return (
            "Pear/triangle-style",
            "Shoulder width appears smaller than hip width in this image; this is a styling proxy.",
        )
    if ratio > 1.10:
        return (
            "Inverted-triangle-style",
            "Shoulder width appears larger than hip width in this image; this is a styling proxy.",
        )
    return (
        "Balanced shoulder/hip-style",
        "Shoulder and hip widths appear similar in this image; this is a styling proxy.",
    )


class BodyShapeEstimator:
    """Run optional MediaPipe Pose on an image held only in process memory."""

    def estimate(self, image_bytes: bytes) -> BodyShapeEstimate:
        if not image_bytes:
            raise BodyShapeError("Choose an image before estimating styling proportions")
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise BodyShapeError("Image must be 5 MB or smaller")
        try:
            import cv2
            import mediapipe as mp
        except ImportError as error:
            raise BodyShapeUnavailable(
                "Body-shape estimation needs MediaPipe and OpenCV. Install project "
                "requirements, then restart the app."
            ) from error

        encoded = np.frombuffer(image_bytes, dtype=np.uint8)
        image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise BodyShapeError("The uploaded file is not a readable image")
        image_height, image_width = image_bgr.shape[:2]
        if image_width * image_height > MAX_IMAGE_PIXELS:
            raise BodyShapeError("Image dimensions are too large; use an image under 20 megapixels")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        try:
            with mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                enable_segmentation=True,
                min_detection_confidence=0.55,
            ) as pose:
                result = pose.process(image_rgb)
        except Exception as error:
            raise BodyShapeError(
                f"Pose processing failed ({type(error).__name__}); try another image."
            ) from error
        if result.pose_landmarks is None:
            raise BodyShapeError(
                "No person pose was detected. Use one front-facing person with the full body visible."
            )

        measurements = measure_pose(
            result.pose_landmarks.landmark,
            image_width,
            image_height,
            result.segmentation_mask,
        )
        category, explanation = classify_styling_shape(measurements)

        annotated_rgb = image_rgb.copy()
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_rgb,
            result.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
        )
        success, encoded_image = cv2.imencode(
            ".jpg",
            cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 80],
        )
        if not success:
            raise BodyShapeError("Could not render the detected pose landmarks")
        image_uri = "data:image/jpeg;base64," + base64.b64encode(
            encoded_image.tobytes()
        ).decode("ascii")
        return BodyShapeEstimate(
            category=category,
            measurements={
                key: round(float(value), 3) if value is not None else None
                for key, value in measurements.items()
            },
            explanation=explanation,
            limitation=(
                "Experimental styling estimate only—not a medical assessment or objective "
                "body-type determination. Ratios vary with camera angle, pose, clothing, "
                "occlusion, and segmentation. Waist is only a silhouette-width proxy when "
                "available. Hourglass and oval labels are not inferred."
            ),
            landmark_image_data_uri=image_uri,
        )
