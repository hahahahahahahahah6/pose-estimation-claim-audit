"""Fixed specification oracles and behavioral regressions."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from pose_estimation_claim_audit.audit import ROUTES, audit, has_evidence, main, match_phrases
from pose_estimation_claim_audit.phrases import POSEESTIMATION, NOPOSEESTIMATION, GENERIC_NAMES

EXPECTED_POSITIVE = ['pose estimation', 'pose-estimation', 'poseestimation', 'human pose estimation', 'human-pose-estimation', 'body pose estimation', 'body-pose-estimation', 'pose detection', 'pose-detection', 'human pose detection', 'body pose detection', 'estimate pose', 'estimates pose', 'estimating pose', 'estimate poses', 'estimates poses', 'estimating poses', 'estimate human pose', 'estimate body pose', 'detect pose', 'detects pose', 'detecting pose', 'detect poses', 'detects poses', 'detecting poses', 'pose estimator', 'pose estimators', 'pose estimation tool', 'pose estimation library', 'pose estimation feature', 'pose estimation support', 'supports pose estimation', 'pose estimation capability', 'pose estimation pipeline', 'pose estimation algorithm', 'pose estimation software', 'pose estimation app', 'pose estimation model', 'keypoint detection', 'keypoint-detection', 'keypoints detection', 'detect keypoints', 'detects keypoints', 'detecting keypoints', 'estimate keypoints', 'estimates keypoints', 'estimating keypoints', 'pose landmarks', 'body keypoints', 'skeleton tracking']
EXPECTED_NEGATIVE = ['does not support pose estimation', 'do not support pose estimation', "doesn't support pose estimation", "don't support pose estimation", 'no pose estimation', 'without pose estimation', 'never performs pose estimation', 'does not perform pose estimation', 'pose estimation not supported', 'pose estimation unsupported', 'no pose estimation support', 'lacks pose estimation', 'missing pose estimation', 'pose estimation disabled', 'no pose estimation capability', 'without pose estimation support', 'does not provide pose estimation', 'no pose estimation model', 'pose estimation not available', 'does not use pose estimation', 'no pose detection', 'without pose detection', 'does not support pose detection', 'no pose estimation feature']
EXPECTED_GENERIC = ['estimate_pose', 'detect_pose', 'predict_pose', 'get_pose', 'estimate_poses', 'detect_poses', 'estimate_keypoints', 'detect_keypoints', 'predict_keypoints', 'get_keypoints', 'estimate_pose_landmarks', 'detect_pose_landmarks', 'find_keypoints', 'extract_keypoints', 'estimate_body_pose', 'detect_body_pose', 'predict_body_pose', 'estimate_human_pose', 'detect_human_pose', 'run_pose_estimation', 'apply_pose_estimation', 'perform_pose_estimation', 'pose_estimation', 'estimate_skeleton']

class AuditTests(unittest.TestCase):
    def test_phrase_oracle(self):
        self.assertEqual(POSEESTIMATION, EXPECTED_POSITIVE)
        self.assertEqual(NOPOSEESTIMATION, EXPECTED_NEGATIVE)
        self.assertEqual(len(POSEESTIMATION), 50)
        self.assertEqual(len(NOPOSEESTIMATION), 24)

    def test_generic_oracle(self):
        self.assertEqual(GENERIC_NAMES, EXPECTED_GENERIC)
        self.assertEqual(len(GENERIC_NAMES), 24)

    def test_route_oracle(self):
        self.assertEqual(ROUTES, (
            {'names': ['Pose'], 'source': 'mediapipe.solutions.pose', 'methods': ['process'], 'direct': False},
            {'names': ['PoseLandmarker'], 'source': 'mediapipe.tasks.python.vision', 'factory': 'create_from_options', 'methods': ['detect', 'detect_for_video', 'detect_async'], 'direct': False},
            {'names': ['YOLO'], 'source': 'ultralytics', 'marker': 'pose', 'argument': 'model', 'methods': ['predict'], 'direct': True, 'excluded': ['train', 'val', 'export', 'track']},
            {'names': ['keypointrcnn_resnet50_fpn', 'KeypointRCNN'], 'source': 'torchvision.models.detection', 'methods': [], 'direct': True},
            {'names': ['load'], 'source': 'tensorflow_hub', 'marker': 'movenet', 'argument': 'first', 'methods': [], 'direct': True, 'containment': True},
            {'names': ['readNetFromCaffe'], 'source': 'cv2.dnn', 'marker': 'pose', 'argument': 'any', 'methods': ['forward'], 'direct': False, 'excluded': ['setInput']},
        ))

    def fixture(self, docs='', source='', test=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        (root / 'README.md').write_text(docs)
        if source:
            (root / 'app.py').write_text(source)
        if test is not None:
            (root / 'tests').mkdir()
            (root / 'tests/fixture.py').write_text(test)
        return root

    def cli(self, args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(args)
        return code, output.getvalue()

    def test_span_removal(self):
        self.assertEqual(match_phrases('does not support pose estimation'), (['does not support pose estimation'], []))

    def test_longest_model(self):
        self.assertEqual(match_phrases('pose estimation model'), ([], ['pose estimation model']))

    def test_longest_human(self):
        self.assertEqual(match_phrases('human pose estimation'), ([], ['human pose estimation']))

    def test_repeated_spans(self):
        self.assertEqual(match_phrases('pose estimation model; pose estimation model')[1], ['pose estimation model'] * 2)

    def test_case(self):
        root = self.fixture('POSE ESTIMATION')
        self.assertEqual(audit(root)[0][0]['severity'], 'HIGH')

    def test_camera_collision(self):
        self.assertEqual(match_phrases('camera pose estimation recovers the camera position')[1], ['pose estimation'])

    def test_face_collision(self):
        self.assertEqual(match_phrases('we detect keypoints on faces')[1], ['detect keypoints'])

    def test_fenced(self):
        root = self.fixture('```text\npose estimation\n```')
        self.assertEqual(audit(root)[0], [])

    def test_inline(self):
        root = self.fixture('`pose estimation`')
        self.assertEqual(audit(root)[0][0]['severity'], 'HIGH')

    def test_no_normalization(self):
        self.assertEqual(match_phrases('pose--estimation'), ([], []))

    def test_good_cli(self):
        root = self.fixture('pose estimation', 'x.estimate_pose(image)')
        code, output = self.cli(['audit', str(root)])
        self.assertEqual(code, 0)
        self.assertIn('PASS', output)

    def test_bad_cli(self):
        root = self.fixture('pose estimation; detect keypoints')
        code, output = self.cli(['audit', str(root), '--json'])
        self.assertEqual(code, 2)
        findings = json.loads(output)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]['rule'], 'documented-pose-estimation-claim-false')
        self.assertEqual(findings[0]['severity'], 'HIGH')
        self.assertEqual(findings[0]['verdict'], 'FAIL')
        self.assertTrue(findings[0]['message'])

    def test_denial_cli(self):
        root = self.fixture('does not support pose estimation', 'x.estimate_pose(image)')
        code, output = self.cli(['audit', str(root), '--json'])
        self.assertEqual(code, 2)
        findings = json.loads(output)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]['rule'], 'documented-no-pose-estimation-claim-false')
        self.assertEqual(findings[0]['severity'], 'HIGH')

    def test_denial_no_source(self):
        self.assertEqual(audit(self.fixture('no pose estimation'))[0], [])

    def test_no_claim_cli(self):
        code, output = self.cli(['audit', str(self.fixture('Hello'))])
        self.assertEqual(code, 0)
        self.assertIn('PASS', output)

    def test_test_only_cli(self):
        root = self.fixture(test='"pose estimation"')
        code, output = self.cli(['audit', str(root), '--json'])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output)[0]['severity'], 'MEDIUM')
        self.assertEqual(self.cli(['audit', str(root), '--fail-on', 'medium'])[0], 2)

    def test_ignore(self):
        root = self.fixture('pose estimation')
        self.assertEqual(self.cli(['audit', str(root), '--ignore', 'READ'])[0], 0)

    def test_ignore_before_positional(self):
        root = self.fixture('pose estimation')
        self.assertEqual(self.cli(['audit', '--ignore', 'READ', str(root)])[0], 0)

    def test_ignore_repeatable(self):
        root = self.fixture('pose estimation', test='"pose estimation"')
        self.assertEqual(self.cli(['audit', '--ignore', 'README', '--ignore', 'tests/', str(root)])[0], 0)

    def test_version(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as cm:
            main(['--version'])
        self.assertEqual(cm.exception.code, 0)
        self.assertEqual(output.getvalue().strip(), '0.1.0')

    def test_readme_self_audit(self):
        readme = Path(__file__).resolve().parents[1] / 'README.md'
        root = self.fixture(readme.read_text())
        self.assertEqual(audit(root)[0], [])
        from pose_estimation_claim_audit.audit import prose
        self.assertEqual(match_phrases(prose(readme.read_text())), ([], []))

    def test_split_modules(self):
        root = self.fixture('pose estimation', 'from mediapipe.solutions.pose import Pose\np = Pose()')
        (root / 'other.py').write_text('p.process(image)')
        self.assertEqual(audit(root)[0][0]['severity'], 'HIGH')

    def test_docs_scope(self):
        root = self.fixture()
        (root / 'docs/nested').mkdir(parents=True)
        (root / 'docs/nested/guide.md').write_text('pose estimation')
        self.assertEqual(audit(root)[0][0]['severity'], 'HIGH')

    def test_outside_docs(self):
        root = self.fixture()
        (root / 'other.md').write_text('pose estimation')
        self.assertEqual(audit(root)[0], [])

    def test_excluded_sources(self):
        root = self.fixture('pose estimation')
        for directory in ['.git', '__pycache__', '.venv', 'custom_env']:
            (root / directory).mkdir()
            (root / directory / 'code.py').write_text('x.estimate_pose()')
        (root / 'custom_env/pyvenv.cfg').write_text('home = /tmp')
        self.assertEqual(audit(root)[0][0]['severity'], 'HIGH')

    def test_verbose(self):
        root = self.fixture('Hello')
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(self.cli(['audit', str(root), '--verbose'])[0], 0)
        self.assertIn('Scope:', err.getvalue())
        self.assertIn('scan README.md', err.getvalue())

    def test_usage_and_io(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
            main(['audit', '--fail-on', 'low'])
        self.assertEqual(cm.exception.code, 1)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['audit', '/nonexistent-audit-directory']), 1)

    def test_evidence_accept_pose_bare(self):
        self.assertTrue(has_evidence('from mediapipe.solutions.pose import Pose\np=Pose()\np.process(image)'))

    def test_evidence_accept_pose_module(self):
        self.assertTrue(has_evidence('from mediapipe.solutions import pose\np=pose.Pose()\np.process(image)'))

    def test_evidence_accept_pose_mp(self):
        self.assertTrue(has_evidence('import mediapipe as mp\np=mp.solutions.pose.Pose()\np.process(image)'))

    def test_evidence_accept_pose_solutions(self):
        self.assertTrue(has_evidence('from mediapipe import solutions\np=solutions.pose.Pose()\np.process(image)'))

    def test_evidence_accept_pose_alias(self):
        self.assertTrue(has_evidence('from mediapipe.solutions.pose import Pose as P\np=P()\np.process(image)'))

    def test_evidence_accept_landmarker_bare(self):
        self.assertTrue(has_evidence('from mediapipe.tasks.python.vision import PoseLandmarker\np=PoseLandmarker.create_from_options(opts)\np.detect(image)'))

    def test_evidence_accept_landmarker_vision(self):
        self.assertTrue(has_evidence('from mediapipe.tasks.python import vision\np=vision.PoseLandmarker.create_from_options(opts)\np.detect_for_video(image,0)'))

    def test_evidence_accept_landmarker_alias(self):
        self.assertTrue(has_evidence('from mediapipe.tasks.python.vision import PoseLandmarker as PL\np=PL.create_from_options(opts)\np.detect_async(image,0)'))

    def test_evidence_accept_landmarker_tasks(self):
        self.assertTrue(has_evidence('from mediapipe import tasks\np=tasks.python.vision.PoseLandmarker.create_from_options(opts)\np.detect(image)'))

    def test_evidence_accept_landmarker_python(self):
        self.assertTrue(has_evidence('from mediapipe.tasks import python\np=python.vision.PoseLandmarker.create_from_options(opts)\np.detect(image)'))

    def test_evidence_accept_yolo_bare(self):
        self.assertTrue(has_evidence('from ultralytics import YOLO\np=YOLO("yolo11n-pose.pt")\np(image)'))

    def test_evidence_accept_yolo_keyword(self):
        self.assertTrue(has_evidence('from ultralytics import YOLO\np=YOLO(model="yolo26n-pose.pt")\np.predict(image)'))

    def test_evidence_accept_yolo_alias(self):
        self.assertTrue(has_evidence('from ultralytics import YOLO as Y\np=Y("POSE.pt")\np(image)'))

    def test_evidence_accept_yolo_module(self):
        self.assertTrue(has_evidence('import ultralytics\np=ultralytics.YOLO("pose.pt")\np(image)'))

    def test_evidence_accept_yolo_collision(self):
        self.assertTrue(has_evidence('from ultralytics import YOLO\np=YOLO("purpose.pt")\np(image)'))

    def test_evidence_accept_rcnn_bare(self):
        self.assertTrue(has_evidence('from torchvision.models.detection import keypointrcnn_resnet50_fpn\np=keypointrcnn_resnet50_fpn()\np(image)'))

    def test_evidence_accept_rcnn_class(self):
        self.assertTrue(has_evidence('from torchvision.models.detection import KeypointRCNN\np=KeypointRCNN()\np(image)'))

    def test_evidence_accept_rcnn_alias(self):
        self.assertTrue(has_evidence('from torchvision.models.detection import keypointrcnn_resnet50_fpn as K\np=K()\np(image)'))

    def test_evidence_accept_rcnn_detection(self):
        self.assertTrue(has_evidence('from torchvision.models import detection\np=detection.KeypointRCNN()\np(image)'))

    def test_evidence_accept_rcnn_models(self):
        self.assertTrue(has_evidence('from torchvision import models\np=models.detection.keypointrcnn_resnet50_fpn()\np(image)'))

    def test_evidence_accept_movenet_hub(self):
        self.assertTrue(has_evidence('import tensorflow_hub as hub\np=hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")\np(image)'))

    def test_evidence_accept_movenet_module(self):
        self.assertTrue(has_evidence('import tensorflow_hub\np=tensorflow_hub.load("MOVENET")\np(image)'))

    def test_evidence_accept_movenet_bare(self):
        self.assertTrue(has_evidence('from tensorflow_hub import load\np=load("movenet")\np(image)'))

    def test_evidence_accept_movenet_alias(self):
        self.assertTrue(has_evidence('from tensorflow_hub import load as L\np=L("movenet")\np(image)'))

    def test_evidence_accept_movenet_export(self):
        self.assertTrue(has_evidence('from tensorflow_hub import hub\np=hub.load("movenet")\np(image)'))

    def test_evidence_accept_movenet_signature(self):
        self.assertTrue(has_evidence('import tensorflow_hub as hub\nmodule=hub.load("movenet")\nmovenet=module.signatures["serving_default"]\nmovenet(image)'))

    def test_evidence_accept_movenet_containment(self):
        self.assertTrue(has_evidence('import tensorflow_hub as hub\nmodule=hub.load("movenet")\nregistry[module](image)'))

    def test_evidence_accept_openpose_cv2(self):
        self.assertTrue(has_evidence('import cv2\np=cv2.dnn.readNetFromCaffe("pose/coco/pose_deploy_linevec.prototxt", weights)\np.forward()'))

    def test_evidence_accept_openpose_dnn(self):
        self.assertTrue(has_evidence('from cv2 import dnn\np=dnn.readNetFromCaffe("POSE.prototxt")\np.forward()'))

    def test_evidence_accept_openpose_alias(self):
        self.assertTrue(has_evidence('import cv2 as cv\np=cv.dnn.readNetFromCaffe("pose.prototxt")\np.forward()'))

    def test_evidence_accept_openpose_weights(self):
        self.assertTrue(has_evidence('import cv2\np=cv2.dnn.readNetFromCaffe("config", weights="pose.caffemodel")\np.forward()'))

    def test_evidence_accept_generic_attribute(self):
        self.assertTrue(has_evidence('unrelated.estimate_pose(image)'))

    def test_evidence_accept_generic_bare(self):
        self.assertTrue(has_evidence('from custom import estimate_pose\nestimate_pose(image)'))

    def test_evidence_accept_generic_alias(self):
        self.assertTrue(has_evidence('from custom import estimate_pose as ep\nep(image)'))

    def test_evidence_accept_load_shadow_attribute_survives(self):
        self.assertTrue(has_evidence('import tensorflow_hub as hub\ndef load(): pass\np=hub.load("movenet")\np(image)'))

    def test_evidence_reject_pose_no_import(self):
        self.assertFalse(has_evidence('p=Pose()\np.process(image)'))

    def test_evidence_reject_pose_wrong_source(self):
        self.assertFalse(has_evidence('from fake import Pose\np=Pose()\np.process(image)'))

    def test_evidence_reject_pose_wrong_base(self):
        self.assertFalse(has_evidence('p=fake.Pose()\np.process(image)'))

    def test_evidence_reject_pose_construct_only(self):
        self.assertFalse(has_evidence('from mediapipe.solutions.pose import Pose\np=Pose()'))

    def test_evidence_reject_pose_process_only(self):
        self.assertFalse(has_evidence('import mediapipe\np.process(image)'))

    def test_evidence_reject_pose_wrong_instance(self):
        self.assertFalse(has_evidence('from mediapipe.solutions.pose import Pose\np=Pose()\nq.process(image)'))

    def test_evidence_reject_landmarker_no_import(self):
        self.assertFalse(has_evidence('p=PoseLandmarker.create_from_options(o)\np.detect(image)'))

    def test_evidence_reject_landmarker_wrong_base(self):
        self.assertFalse(has_evidence('p=fake.PoseLandmarker.create_from_options(o)\np.detect(image)'))

    def test_evidence_reject_landmarker_construct_only(self):
        self.assertFalse(has_evidence('from mediapipe.tasks.python.vision import PoseLandmarker\np=PoseLandmarker.create_from_options(o)'))

    def test_evidence_reject_landmarker_detect_only(self):
        self.assertFalse(has_evidence('import mediapipe\np.detect(image)'))

    def test_evidence_reject_yolo_detector(self):
        self.assertFalse(has_evidence('from ultralytics import YOLO\np=YOLO("yolov8n.pt")\np(image)'))

    def test_evidence_reject_yolo_variable(self):
        self.assertFalse(has_evidence('from ultralytics import YOLO\np=YOLO(path)\np(image)'))

    def test_evidence_reject_yolo_no_import(self):
        self.assertFalse(has_evidence('p=YOLO("pose.pt")\np(image)'))

    def test_evidence_reject_yolo_wrong_base(self):
        self.assertFalse(has_evidence('p=fake.YOLO("pose.pt")\np(image)'))

    def test_evidence_reject_yolo_construct_only(self):
        self.assertFalse(has_evidence('from ultralytics import YOLO\np=YOLO("pose.pt")'))

    def test_evidence_reject_rcnn_no_import(self):
        self.assertFalse(has_evidence('p=keypointrcnn_resnet50_fpn()\np(image)'))

    def test_evidence_reject_rcnn_class_no_import(self):
        self.assertFalse(has_evidence('p=KeypointRCNN()\np(image)'))

    def test_evidence_reject_rcnn_wrong_base(self):
        self.assertFalse(has_evidence('p=fake.KeypointRCNN()\np(image)'))

    def test_evidence_reject_rcnn_construct_only(self):
        self.assertFalse(has_evidence('from torchvision.models.detection import KeypointRCNN\np=KeypointRCNN()'))

    def test_evidence_reject_rcnn_call_only(self):
        self.assertFalse(has_evidence('from torchvision.models.detection import KeypointRCNN\np(image)'))

    def test_evidence_reject_movenet_wrong_model(self):
        self.assertFalse(has_evidence('import tensorflow_hub as hub\np=hub.load("https://tfhub.dev/google/movinet/a0")\np(image)'))

    def test_evidence_reject_movenet_variable(self):
        self.assertFalse(has_evidence('import tensorflow_hub as hub\np=hub.load(path)\np(image)'))

    def test_evidence_reject_movenet_bare_no_import(self):
        self.assertFalse(has_evidence('p=load("movenet")\np(image)'))

    def test_evidence_reject_movenet_load_only(self):
        self.assertFalse(has_evidence('import tensorflow_hub as hub\np=hub.load("movenet")'))

    def test_evidence_reject_movenet_call_only(self):
        self.assertFalse(has_evidence('import tensorflow_hub as hub\np(image)'))

    def test_evidence_reject_movenet_keyword_rejected(self):
        self.assertFalse(has_evidence('import tensorflow_hub as hub\np=hub.load(handle="movenet")\np(image)'))

    def test_evidence_reject_openpose_face(self):
        self.assertFalse(has_evidence('import cv2\np=cv2.dnn.readNetFromCaffe("face.prototxt", "face.caffemodel")\np.forward()'))

    def test_evidence_reject_openpose_wrong_base(self):
        self.assertFalse(has_evidence('p=other.readNetFromCaffe("pose.prototxt")\np.forward()'))

    def test_evidence_reject_openpose_setinput(self):
        self.assertFalse(has_evidence('import cv2\np=cv2.dnn.readNetFromCaffe("pose.prototxt")\np.setInput(image)'))

    def test_evidence_reject_openpose_construct_only(self):
        self.assertFalse(has_evidence('import cv2\np=cv2.dnn.readNetFromCaffe("pose.prototxt")'))

    def test_evidence_reject_openpose_forward_only(self):
        self.assertFalse(has_evidence('import cv2\np.forward()'))

    def test_evidence_reject_openpose_bare(self):
        self.assertFalse(has_evidence('from cv2.dnn import readNetFromCaffe\np=readNetFromCaffe("pose.prototxt")\np.forward()'))

    def test_evidence_reject_bare_imports(self):
        self.assertFalse(has_evidence('import mediapipe, ultralytics, torchvision, tensorflow_hub, cv2'))

    def test_evidence_reject_strings_comments(self):
        self.assertFalse(has_evidence('"""pose estimation"""\n# detect_pose()\ntext="keypoint"'))

    def test_evidence_reject_unbound(self):
        self.assertFalse(has_evidence('from custom import estimate_pose\nestimate_pose'))

    def test_evidence_reject_generic_unimported(self):
        self.assertFalse(has_evidence('estimate_pose(image)'))

    def test_evidence_reject_subprocess(self):
        self.assertFalse(has_evidence('import subprocess\nsubprocess.run(["pose-estimation", "image.jpg"])'))

    def test_evidence_reject_cascade(self):
        self.assertFalse(has_evidence('import cv2\np=cv2.CascadeClassifier("face.xml")\np.detectMultiScale(image)'))

    def test_evidence_reject_background(self):
        self.assertFalse(has_evidence('from rembg import remove\nremove(image)'))

    def test_evidence_reject_depth(self):
        self.assertFalse(has_evidence('depth_model.estimate_depth(image)'))

    def test_evidence_reject_flow(self):
        self.assertFalse(has_evidence('cv2.calcOpticalFlowFarneback(a,b)'))

    def test_evidence_reject_classification(self):
        self.assertFalse(has_evidence('classifier.predict(image)'))

    def test_evidence_reject_video(self):
        self.assertFalse(has_evidence('video_classifier.predict(video)'))

    def test_evidence_reject_caption(self):
        self.assertFalse(has_evidence('captioner.generate(image)'))

    def test_evidence_reject_generation(self):
        self.assertFalse(has_evidence('generator.generate(prompt)'))

    def test_evidence_reject_action(self):
        self.assertFalse(has_evidence('action_model.recognize(joints)'))

    def test_evidence_reject_gait(self):
        self.assertFalse(has_evidence('gait_model.recognize(joints)'))

    def test_evidence_reject_yolo_train(self):
        self.assertFalse(has_evidence('from ultralytics import YOLO\np=YOLO("pose.pt")\np.train()'))

    def test_evidence_reject_yolo_val(self):
        self.assertFalse(has_evidence('from ultralytics import YOLO\np=YOLO("pose.pt")\np.val()'))

    def test_evidence_reject_yolo_export(self):
        self.assertFalse(has_evidence('from ultralytics import YOLO\np=YOLO("pose.pt")\np.export()'))

    def test_evidence_reject_yolo_track(self):
        self.assertFalse(has_evidence('from ultralytics import YOLO\np=YOLO("pose.pt")\np.track()'))

    def test_evidence_reject_fasterrcnn_resnet50_fpn(self):
        self.assertFalse(has_evidence('from torchvision.models.detection import fasterrcnn_resnet50_fpn\np=fasterrcnn_resnet50_fpn()\np(image)'))

    def test_evidence_reject_retinanet_resnet50_fpn(self):
        self.assertFalse(has_evidence('from torchvision.models.detection import retinanet_resnet50_fpn\np=retinanet_resnet50_fpn()\np(image)'))

    def test_evidence_reject_ssd300_vgg16(self):
        self.assertFalse(has_evidence('from torchvision.models.detection import ssd300_vgg16\np=ssd300_vgg16()\np(image)'))

    def test_evidence_reject_maskrcnn_resnet50_fpn(self):
        self.assertFalse(has_evidence('from torchvision.models.detection import maskrcnn_resnet50_fpn\np=maskrcnn_resnet50_fpn()\np(image)'))

    def test_evidence_reject_FaceMesh(self):
        self.assertFalse(has_evidence('import mediapipe as mp\np=mp.solutions.other.FaceMesh()\np.process(image)'))

    def test_evidence_reject_Hands(self):
        self.assertFalse(has_evidence('import mediapipe as mp\np=mp.solutions.other.Hands()\np.process(image)'))

    def test_evidence_reject_FaceDetection(self):
        self.assertFalse(has_evidence('import mediapipe as mp\np=mp.solutions.other.FaceDetection()\np.process(image)'))

    def test_evidence_reject_SelfieSegmentation(self):
        self.assertFalse(has_evidence('import mediapipe as mp\np=mp.solutions.other.SelfieSegmentation()\np.process(image)'))

    def test_evidence_reject_shadow_Pose_function(self):
        self.assertFalse(has_evidence('from mediapipe.solutions.pose import Pose\np=Pose()\np.process(image)\ndef Pose(): pass'))

    def test_evidence_reject_shadow_Pose_class(self):
        self.assertFalse(has_evidence('from mediapipe.solutions.pose import Pose\np=Pose()\np.process(image)\nclass Pose: pass'))

    def test_evidence_reject_shadow_PoseLandmarker_function(self):
        self.assertFalse(has_evidence('from mediapipe.tasks.python.vision import PoseLandmarker\np=PoseLandmarker.create_from_options(opts)\np.detect(image)\ndef PoseLandmarker(): pass'))

    def test_evidence_reject_shadow_PoseLandmarker_class(self):
        self.assertFalse(has_evidence('from mediapipe.tasks.python.vision import PoseLandmarker\np=PoseLandmarker.create_from_options(opts)\np.detect(image)\nclass PoseLandmarker: pass'))

    def test_evidence_reject_shadow_YOLO_function(self):
        self.assertFalse(has_evidence('from ultralytics import YOLO\np=YOLO("yolo11n-pose.pt")\np(image)\ndef YOLO(): pass'))

    def test_evidence_reject_shadow_YOLO_class(self):
        self.assertFalse(has_evidence('from ultralytics import YOLO\np=YOLO("yolo11n-pose.pt")\np(image)\nclass YOLO: pass'))

    def test_evidence_reject_shadow_keypointrcnn_resnet50_fpn_function(self):
        self.assertFalse(has_evidence('from torchvision.models.detection import keypointrcnn_resnet50_fpn\np=keypointrcnn_resnet50_fpn()\np(image)\ndef keypointrcnn_resnet50_fpn(): pass'))

    def test_evidence_reject_shadow_keypointrcnn_resnet50_fpn_class(self):
        self.assertFalse(has_evidence('from torchvision.models.detection import keypointrcnn_resnet50_fpn\np=keypointrcnn_resnet50_fpn()\np(image)\nclass keypointrcnn_resnet50_fpn: pass'))

    def test_evidence_reject_shadow_KeypointRCNN_function(self):
        self.assertFalse(has_evidence('from torchvision.models.detection import KeypointRCNN\np=KeypointRCNN()\np(image)\ndef KeypointRCNN(): pass'))

    def test_evidence_reject_shadow_KeypointRCNN_class(self):
        self.assertFalse(has_evidence('from torchvision.models.detection import KeypointRCNN\np=KeypointRCNN()\np(image)\nclass KeypointRCNN: pass'))

    def test_evidence_reject_shadow_load_function(self):
        self.assertFalse(has_evidence('from tensorflow_hub import load\np=load("movenet")\np(image)\ndef load(): pass'))

    def test_evidence_reject_shadow_load_class(self):
        self.assertFalse(has_evidence('from tensorflow_hub import load\np=load("movenet")\np(image)\nclass load: pass'))

    def test_evidence_reject_shadow_estimate_pose_function(self):
        self.assertFalse(has_evidence('from custom import estimate_pose\nestimate_pose(image)\ndef estimate_pose(): pass'))

    def test_evidence_reject_shadow_estimate_pose_class(self):
        self.assertFalse(has_evidence('from custom import estimate_pose\nestimate_pose(image)\nclass estimate_pose: pass'))

    def test_all_generic_calls(self):
        for name in EXPECTED_GENERIC:
            with self.subTest(name=name):
                self.assertTrue(has_evidence('x.' + name + '(image)'))
                self.assertTrue(has_evidence('from custom import ' + name + ' as alias\nalias(image)'))

    def test_all_phrases(self):
        for phrase in EXPECTED_POSITIVE:
            with self.subTest(phrase=phrase):
                self.assertEqual(match_phrases(phrase), ([], [phrase]))
        for phrase in EXPECTED_NEGATIVE:
            with self.subTest(phrase=phrase):
                self.assertEqual(match_phrases(phrase), ([phrase], []))


if __name__ == '__main__':
    unittest.main()
