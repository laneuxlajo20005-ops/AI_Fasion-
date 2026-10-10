# Optional pose-based styling estimate

## Pipeline and data handling

An optional uploaded image or in-browser webcam capture is read into memory,
decoded by OpenCV, and passed to MediaPipe Pose. The server draws detected pose
landmarks on a response-only preview. In the guided wizard, neither the
original image nor annotated preview is persisted; only the estimated label,
measurements, explanation, and limitation remain in server-side wizard state
for up to 24 hours. The legacy profile API may persist a selected styling label
as a preference, but it does not save an image or numeric measurements. Start
Over deletes the current wizard record.

Upload size is limited to 5 MB and decoded images to 20 megapixels. Only one
image is accepted per request. Webcam access requires browser permission and is
normally available on `localhost` or HTTPS.

## Measurements and supported labels

The estimate requires visible shoulders, hips, and ankles:

- Shoulder and hip widths are horizontal pixel distances between left/right
  pose landmarks.
- Shoulder-to-hip width ratio compares those two widths.
- Shoulder-width/torso-length and hip-width/torso-length normalize widths by
  shoulder-to-hip landmark distance.
- Torso/leg proportions use shoulder-to-hip and hip-to-ankle landmark distances.
- If MediaPipe segmentation is available, a horizontal silhouette width near
  the midpoint of shoulder-to-hip height is divided by hip width. This is only a
  waist-location silhouette proxy, not a waist measurement.

The rule-based labels are `Pear/triangle-style` below a shoulder-to-hip ratio
of 0.90, `Inverted-triangle-style` above 1.10, and
`Balanced shoulder/hip-style` between those thresholds. They are styling
proxies for recommendation experiments, not medical, objective, or validated
body-type classifications. No labeled body-shape dataset exists in the
project, so no supervised classifier is trained. Hourglass and oval are not
estimated because the available pose landmarks do not reliably measure waist
shape or torso circumference.

Perspective, clothing, pose, occlusion, camera distance, segmentation errors,
and person orientation can all change these 2D proportions. Users can omit the
image and use the preference controls alone. If detection fails or optional
dependencies are unavailable, the app continues recommending without a new
body-shape estimate.

The current MediaPipe Pose pipeline processes one detected pose and does not
reliably count or reject multiple people in a group image. The interface asks
for a single front-facing person; group-photo detection/rejection remains
unimplemented and must not be treated as supported behavior.

## Recommendation relationship

The inferred styling proxy is mapped to category groups (for example, a
Pear/triangle-style proxy favors Tops, Outerwear, and Accessories). This is a
visible, replaceable stylistic heuristic—not a claim about fit, suitability, or
what a person should wear. A body-shape/category fit contributes 12% of the
pre-weather score, and the existing weather context can still contribute 20%.
The UI shows the measurement ratios, detected landmarks, estimate, and
limitations alongside recommendations.
