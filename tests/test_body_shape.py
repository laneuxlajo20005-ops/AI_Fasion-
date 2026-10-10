import unittest
import cv2
from types import SimpleNamespace

import numpy as np

from body_shape import (
    BODY_SHAPE_LABELS,
    BodyShapeError,
    BodyShapeEstimator,
    classify_styling_shape,
    measure_pose,
)


def synthetic_landmarks(shoulder_width=0.30, hip_width=0.20):
    landmarks = [SimpleNamespace(x=0.5, y=0.5, visibility=1.0) for _ in range(33)]
    for index, x, y in (
        (11, 0.5 - shoulder_width / 2, 0.20),
        (12, 0.5 + shoulder_width / 2, 0.20),
        (23, 0.5 - hip_width / 2, 0.50),
        (24, 0.5 + hip_width / 2, 0.50),
        (27, 0.43, 0.95),
        (28, 0.57, 0.95),
    ):
        landmarks[index] = SimpleNamespace(x=x, y=y, visibility=1.0)
    return landmarks


class BodyShapeMeasurementTests(unittest.TestCase):
    def test_normalized_proportions_and_optional_waist_proxy(self):
        mask = np.zeros((200, 100), dtype=np.float32)
        mask[:, 35:66] = 1.0
        mask[70:78, :] = 0.0
        mask[70:78, 42:58] = 1.0
        measurements = measure_pose(
            synthetic_landmarks(),
            image_width=100,
            image_height=200,
            segmentation_mask=mask,
        )
        self.assertAlmostEqual(measurements["shoulder_to_hip_width_ratio"], 1.5)
        self.assertAlmostEqual(
            measurements["torso_proportion_of_shoulder_to_ankle"]
            + measurements["leg_proportion_of_shoulder_to_ankle"],
            1.0,
        )
        self.assertIsNotNone(measurements["waist_silhouette_proxy_over_hip_width"])

    def test_classifier_only_returns_supported_styling_proxies(self):
        categories = []
        for shoulder_width, hip_width in (
            (0.16, 0.20),
            (0.30, 0.20),
            (0.20, 0.20),
        ):
            measurements = measure_pose(
                synthetic_landmarks(shoulder_width, hip_width),
                100,
                200,
                None,
            )
            categories.append(classify_styling_shape(measurements)[0])
        self.assertEqual(categories, list(BODY_SHAPE_LABELS))

    def test_missing_waist_proxy_does_not_prevent_estimation(self):
        measurements = measure_pose(
            synthetic_landmarks(),
            image_width=100,
            image_height=200,
            segmentation_mask=None,
        )
        self.assertIsNone(measurements["waist_silhouette_proxy_over_hip_width"])

    def test_unclear_landmarks_are_rejected(self):
        landmarks = synthetic_landmarks()
        landmarks[11].visibility = 0.1
        with self.assertRaisesRegex(BodyShapeError, "landmarks are unclear"):
            measure_pose(landmarks, 100, 200, None)

    def test_real_mediapipe_pipeline_rejects_image_without_person_pose(self):
        image = np.full((240, 160, 3), 255, dtype=np.uint8)
        success, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(success)
        with self.assertRaisesRegex(BodyShapeError, "No person pose was detected"):
            BodyShapeEstimator().estimate(encoded.tobytes())


if __name__ == "__main__":
    unittest.main()
