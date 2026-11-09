"""Offline documentation checks and module-local AST evidence detection."""
import argparse
import ast
import json
from pathlib import Path
import re
import sys

from .phrases import POSEESTIMATION, NOPOSEESTIMATION, GENERIC_NAMES

ROUTES = (
    {'names': ['Pose'], 'source': 'mediapipe.solutions.pose', 'methods': ['process'], 'direct': False},
    {'names': ['PoseLandmarker'], 'source': 'mediapipe.tasks.python.vision', 'factory': 'create_from_options', 'methods': ['detect', 'detect_for_video', 'detect_async'], 'direct': False},
    {'names': ['YOLO'], 'source': 'ultralytics', 'marker': 'pose', 'argument': 'model', 'methods': ['predict'], 'direct': True, 'excluded': ['train', 'val', 'export', 'track']},
    {'names': ['keypointrcnn_resnet50_fpn', 'KeypointRCNN'], 'source': 'torchvision.models.detection', 'methods': [], 'direct': True},
    {'names': ['load'], 'source': 'tensorflow_hub', 'marker': 'movenet', 'argument': 'first', 'methods': [], 'direct': True, 'containment': True},
    {'names': ['readNetFromCaffe'], 'source': 'cv2.dnn', 'marker': 'pose', 'argument': 'any', 'methods': ['forward'], 'direct': False, 'excluded': ['setInput']},
)
EXCLUDED = {'.git', '__pycache__', '.venv', 'venv', 'env', '.env', 'virtualenv'}


def match_phrases(text):
    """Return denial and positive occurrences, removing longest spans first."""
    text = text.lower()
    groups = []
    for phrases in (NOPOSEESTIMATION, POSEESTIMATION):
        matches = []
        for phrase in sorted(phrases, key=len, reverse=True):
            start = text.find(phrase)
            while start >= 0:
                matches.append(phrase)
                text = text[:start] + ' ' * len(phrase) + text[start + len(phrase):]
                start = text.find(phrase)
        groups.append(matches)
    return tuple(groups)


def prose(text):
    result = []
    fenced = False
    for line in text.splitlines():
        if line.lstrip().startswith('```'):
            fenced = not fenced
        elif not fenced:
            result.append(line.replace('`', ''))
    return '\n'.join(result)


def has_evidence(source):
    """Recognize paired calls without importing or executing inspected code."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return False
    nodes = list(ast.walk(tree))
    shadows = {n.name for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    imports = {}
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports[alias.asname or alias.name.split('.')[0]] = alias.name if alias.asname else alias.name.split('.')[0]
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                imports[alias.asname or alias.name] = node.module + '.' + alias.name
    # The documented hub re-export resolves to the same module.
    imports = {k: ('tensorflow_hub' if v == 'tensorflow_hub.hub' else v) for k, v in imports.items()}

    def resolve(expr):
        if isinstance(expr, ast.Name):
            return '' if expr.id in shadows else imports.get(expr.id, '')
        if isinstance(expr, ast.Attribute):
            base = resolve(expr.value)
            if expr.attr in shadows and not (expr.attr == 'load' and base == 'tensorflow_hub'):
                return ''
            return base + '.' + expr.attr if base else ''
        return ''

    calls = [n for n in nodes if isinstance(n, ast.Call)]
    for call in calls:
        func = call.func
        if isinstance(func, ast.Attribute) and func.attr in GENERIC_NAMES and func.attr not in shadows:
            return True
        if isinstance(func, ast.Name) and func.id not in shadows and resolve(func).split('.')[-1] in GENERIC_NAMES:
            return True
    assignments = []
    for n in nodes:
        if isinstance(n, ast.Assign):
            assignments.append((n.targets, n.value))
        elif isinstance(n, ast.AnnAssign) and n.value is not None:
            assignments.append(([n.target], n.value))

    def names(expr):
        return {n.id for n in ast.walk(expr) if isinstance(n, ast.Name)}

    for route in ROUTES:
        constructors = []
        for call in calls:
            path = resolve(call.func)
            expected = [route['source'] + '.' + name + ('.' + route['factory'] if 'factory' in route else '') for name in route['names']]
            if path not in expected:
                continue
            # This route is attribute-only, unlike the bare constructor routes.
            if route['source'] == 'cv2.dnn' and not isinstance(call.func, ast.Attribute):
                continue
            if 'marker' in route:
                args = call.args[:1]
                if route['argument'] == 'model' and not args:
                    args = [kw.value for kw in call.keywords if kw.arg == 'model']
                if route['argument'] == 'any':
                    args = call.args + [kw.value for kw in call.keywords]
                if not any(isinstance(arg, ast.Constant) and isinstance(arg.value, str) and route['marker'] in arg.value.lower() for arg in args):
                    continue
            constructors.append(call)
        bound = set()
        for targets, value in assignments:
            if any(value is call for call in constructors):
                for target in targets:
                    bound.update(names(target) - shadows)
        # Propagate only the explicitly documented name-containment approximation.
        if route.get('containment'):
            while True:
                previous = set(bound)
                for targets, value in assignments:
                    if names(value) & bound:
                        for target in targets:
                            bound.update(names(target) - shadows)
                if previous == bound:
                    break
        for call in calls:
            if route.get('containment') and names(call.func) & bound:
                return True
            if route['direct'] and isinstance(call.func, ast.Name) and call.func.id in bound:
                return True
            if isinstance(call.func, ast.Attribute) and call.func.attr in route['methods']:
                if isinstance(call.func.value, ast.Name) and call.func.value.id in bound:
                    return True
                if any(call.func.value is constructor for constructor in constructors):
                    return True
            if route['direct'] and any(call.func is constructor for constructor in constructors):
                return True
    return False


def audit(repo, ignore=(), verbose=False):
    repo = Path(repo)
    if not repo.is_dir():
        raise OSError('Repository is not a directory: ' + str(repo))
    positives, denials, test_claims, evidence = [], [], [], []
    decisions = []
    for path in sorted(repo.rglob('*')):
        relative = path.relative_to(repo)
        label = relative.as_posix()
        if any(part in EXCLUDED for part in relative.parts) or any((repo.joinpath(*relative.parts[:i]) / 'pyvenv.cfg').is_file() for i in range(1, len(relative.parts))):
            continue
        if any(label.startswith(prefix) for prefix in ignore):
            decisions.append('skip ' + label)
            continue
        if not path.is_file():
            continue
        doc = label == 'README.md' or (relative.parts[0] == 'docs' and path.suffix == '.md')
        test = 'tests' in relative.parts[:-1]
        if doc or test or path.suffix == '.py':
            text = path.read_text(encoding='utf-8')
            decisions.append('scan ' + label)
            if doc or test:
                no, yes = match_phrases(prose(text))
                if doc:
                    positives.extend(yes)
                    denials.extend(no)
                elif test:
                    test_claims.extend(yes)
            if path.suffix == '.py' and has_evidence(text):
                evidence.append(label)
    findings = []
    def finding(rule, severity, message):
        findings.append({'rule': rule, 'severity': severity, 'verdict': 'FAIL', 'message': message})
    if positives and not evidence:
        finding('documented-pose-estimation-claim-false', 'HIGH', 'Documentation claims the checked feature, but no source evidence was found.')
    elif test_claims and not evidence:
        finding('documented-pose-estimation-claim-false', 'MEDIUM', 'Only test fixtures claim the checked feature; no source evidence was found.')
    if denials and evidence:
        finding('documented-no-pose-estimation-claim-false', 'HIGH', 'Documentation denies the checked feature, but source evidence was found.')
    return findings, decisions


class Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(1, self.prog + ': error: ' + message + '\n')


def main(argv=None):
    parser = Parser(prog='pose-estimation-claim-audit')
    parser.add_argument('--version', action='version', version='0.1.0')
    sub = parser.add_subparsers(dest='command', required=True, parser_class=Parser)
    command = sub.add_parser('audit')
    command.add_argument('repo', nargs='?', default='.')
    command.add_argument('--json', action='store_true')
    command.add_argument('--verbose', action='store_true')
    command.add_argument('--ignore', action='append', default=[])
    command.add_argument('--fail-on', choices=['high', 'medium'], default='high')
    args = parser.parse_args(argv)
    try:
        findings, decisions = audit(args.repo, args.ignore)
    except (OSError, UnicodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.verbose:
        print('Scope: README.md, docs/**/*.md, tests/**; evidence: repository *.py', file=sys.stderr)
        for decision in decisions:
            print(decision, file=sys.stderr)
    if args.json:
        print(json.dumps(findings))
    elif findings:
        for item in findings:
            print('{severity} {rule}: {message}'.format(**item))
    else:
        print('PASS: no contradictory claims found.')
    return 2 if any(f['severity'] == 'HIGH' or args.fail_on == 'medium' for f in findings) else 0
