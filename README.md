# Documentation claim auditor

An offline, deterministic auditor for claims about estimating the positions
of body joints in a picture. Uses only the standard library at runtime,
requires Python 3.9 or newer, and performs no network requests, telemetry,
or writes. It inspects source text without executing the inspected code.

## Install and run

```sh
python3 -m pip install .
pose-estimation-claim-audit --version
pose-estimation-claim-audit audit .
python3 -m pose_estimation_claim_audit audit . --json
pose-estimation-claim-audit audit --ignore vendor/ .
pose-estimation-claim-audit audit . --ignore vendor/ --ignore generated/ --fail-on medium --verbose
```

The repository defaults to the current directory. Ignore prefixes are
relative paths and each option takes exactly one value. Verbose output
reports scope and file decisions on standard error. JSON output is an array
of findings, each containing rule, severity, verdict, and message fields.
An empty array means no findings. Text output reports PASS when empty.

Exit 0 means no findings at the selected threshold, exit 2 means findings
at or above it, and exit 1 means usage or input/output error. The default
threshold is HIGH; MEDIUM also fails when explicitly selected.

## Scope and rules

Shipped documentation consists of the root README.md and Markdown files
recursively under docs/. Triple-backtick fenced blocks are excluded;
inline backticks are removed while their content remains prose. Matching
uses case-insensitive exact substrings with no stemming, fuzzy matching,
or hyphen normalization. Denial spans are removed first. Longest-first
matching blanks each matched span, so each occurrence matches exactly once
under its longest phrase. Multiple occurrences collapse into one finding
per rule family.

A positive claim without source evidence is HIGH. A denial contradicted
by source evidence is HIGH. A denial with no evidence passes. Claims only
in files beneath a tests/ directory are MEDIUM when evidence is absent;
they pass the default threshold. With evidence, positive claims pass.
No claims also pass.

```text
documented-pose-estimation-claim-false
documented-no-pose-estimation-claim-false
```

All repository source files with the .py suffix are checked, excluding
version-control metadata, bytecode caches, and virtual environments
(including directories containing pyvenv.cfg). Evidence must be complete
within a single module. Imports are resolved with aliases; a local function
or class shadows the corresponding route name throughout that module.
Imported names alone, comments, strings, uncalled references, subprocess
commands, and pairs split between modules do not count.

## Evidence routes

The following table is kept inside a code block to separate syntax examples
from documentation assertions. The first six routes require construction
and the specified call on its resulting object in the same module.

```text
1. mediapipe.solutions.pose.Pose(...) -> instance.process(...)
2. mediapipe.tasks.python.vision.PoseLandmarker.create_from_options(...)
   -> instance.detect(...) / detect_for_video(...) / detect_async(...)
3. ultralytics.YOLO("yolo11n-pose.pt") -> instance(...) / instance.predict(...)
   First positional argument or model= must be a constant string containing
   "pose", case-insensitively. train/val/export/track alone are rejected.
4. torchvision.models.detection.keypointrcnn_resnet50_fpn(...) or KeypointRCNN(...)
   -> instance(...)
5. tensorflow_hub.load("movenet...") -> instance(...) or signature alias(...)
   First positional argument must be a constant string containing "movenet",
   case-insensitively. Includes from tensorflow_hub import hub/load aliases.
6. cv2.dnn.readNetFromCaffe(...) -> instance.forward(...)
   Any positional/keyword constant string argument must contain "pose",
   case-insensitively. setInput alone is rejected.
7. Attribute calls on any base, or imported/aliased bare calls named:
   estimate_pose, detect_pose, predict_pose, get_pose, estimate_poses,
   detect_poses, estimate_keypoints, detect_keypoints, predict_keypoints,
   get_keypoints, estimate_pose_landmarks, detect_pose_landmarks,
   find_keypoints, extract_keypoints, estimate_body_pose, detect_body_pose,
   predict_body_pose, estimate_human_pose, detect_human_pose,
   run_pose_estimation, apply_pose_estimation, perform_pose_estimation,
   pose_estimation, estimate_skeleton.
```

The first five routes support matching library attribute paths and
from-import aliases. The sixth requires the matching library attribute
path. Bare constructors without the matching imports do not count.

## Differentiation

These neighboring tools check distinct capabilities. Finding and boxing
things in one picture is separate from estimating the positions of body
joints in a picture:

```text
image-object-detection-claim-audit
```

Dividing a picture into labelled regions or masks is separate from
estimating the positions of body joints in a picture:

```text
image-segmentation-claim-audit
```

Estimating how far away things are is separate from estimating the
positions of body joints in a picture:

```text
depth-claim-audit
```

Estimating per-pixel motion between frames is separate from estimating
the positions of body joints in a picture:

```text
optical-flow-claim-audit
```

Naming what a picture shows is separate from estimating the positions
of body joints in a picture:

```text
image-classification-claim-audit
```

Naming what a video shows is separate from estimating the positions
of body joints in a picture:

```text
video-classification-claim-audit
```

The following neighbor checks only region separation through the named
library. This auditor checks that library's body-joint routes and the
other model routes above:

```text
segmentation-claim-audit: mediapipe-based segmentation only
This auditor: mediapipe Pose / PoseLandmarker and the listed model routes
```

Face and hand points, background removal, distance maps, pixel motion,
classification, captioning, generation, action recognition, and gait
recognition alone provide no evidence for the checked capability.

## Limitations

This is static pattern matching, not semantic judgement. Exact phrases
are checked; restatements can be missed. Syntax errors provide no evidence.
There is no cross-module data flow or runtime model validation.

The following matching phrases can also describe camera positioning or
face and hand points, respectively. They still match; those meanings do
not establish the checked capability:

```text
camera pose estimation recovers the camera position
we detect keypoints on faces
```

A model filename can trigger the substring marker accidentally. This
example still counts if followed by the required call:

```text
YOLO("purpose.pt")
```

Likewise, the configuration path in the sixth route may contain its marker
for an unrelated reason and still count:

```text
OpenPose readNetFromCaffe: a prototxt path containing "pose"
```

The fifth route approximates signature indirection by name containment in
the callee. A call that merely mentions a bound name inside its callee
counts. The generic route also accepts shared method names on unrelated
objects:

```text
module = hub.load("movenet")
registry[module](image)
unrelated.estimate_pose(image)
```

## Development

```sh
python3 -m unittest discover
python3 -m build
```

The source archive includes the discoverable test package. The README-only
self-check passes. Scanning this entire project yields a MEDIUM fixture
finding from the fixed phrase-oracle literals in the test suite.

Licensed under MIT.
