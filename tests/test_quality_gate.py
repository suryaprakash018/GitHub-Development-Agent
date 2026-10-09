"""Unit tests for the QualityGate evaluation rules and thresholds."""

from agent.core.config import QualityGateConfig
from agent.selection.models import TaskCandidate, TaskType
from agent.synthesis.models import GeneratedFile, ImplementationPlan
from agent.validation.linter import LintRunResult
from agent.validation.quality_gate import QualityGate
from agent.validation.test_runner import TestRunResult


def test_quality_gate_approves_valid_work(tmp_path):
    gate = QualityGate(QualityGateConfig(minimum_lines_changed=10))

    candidate = TaskCandidate(
        task_id="task_ds_01",
        track_id="track_data_science",
        project_id="proj_ds_01",
        milestone_id="ms_01",
        title="Implement Outliers",
        description="Outlier detector",
        task_type=TaskType.NEW_MODULE,
        target_files=["src/outliers.py"],
        test_files=["tests/test_outliers.py"],
        rationale="Clean implementation",
    )

    code_content = (
        "import numpy as np\n\n"
        "class OutlierDetector:\n"
        "    '''Production outlier detector supporting z-score and IQR methods.'''\n"
        "    def __init__(self, threshold: float = 3.0) -> None:\n"
        "        self.threshold = threshold\n\n"
        "    def detect_zscore(self, values: list[float]) -> list[bool]:\n"
        "        '''Detects outliers using standard score method.'''\n"
        "        arr = np.array(values, dtype=float)\n"
        "        if len(arr) == 0:\n"
        "            return []\n"
        "        mean = np.mean(arr)\n"
        "        std = np.std(arr)\n"
        "        if std == 0:\n"
        "            return [False] * len(arr)\n"
        "        z_scores = np.abs((arr - mean) / std)\n"
        "        return (z_scores > self.threshold).tolist()\n\n"
        "    def detect_iqr(self, values: list[float]) -> list[bool]:\n"
        "        '''Detects outliers using Interquartile Range method.'''\n"
        "        arr = np.array(values, dtype=float)\n"
        "        q25, q75 = np.percentile(arr, [25, 75])\n"
        "        iqr = q75 - q25\n"
        "        lower = q25 - 1.5 * iqr\n"
        "        upper = q75 + 1.5 * iqr\n"
        "        return ((arr < lower) | (arr > upper)).tolist()\n"
    )

    test_content = (
        "from src.outliers import OutlierDetector\n\n"
        "def test_outlier_detection():\n"
        "    detector = OutlierDetector(threshold=2.0)\n"
        "    assert detector.detect([1.0, 3.0]) == [False, True]\n"
    )

    plan = ImplementationPlan(
        task_id="task_ds_01",
        summary="Implement robust outlier detection",
        files=[
            GeneratedFile(
                relative_path="src/outliers.py", content=code_content, description="Module"
            ),
            GeneratedFile(
                relative_path="tests/test_outliers.py", content=test_content, description="Tests"
            ),
        ],
    )

    test_res = TestRunResult(success=True, exit_code=0, passed_count=2, failed_count=0)
    lint_res = LintRunResult(success=True, exit_code=0)

    result = gate.evaluate(candidate, plan, test_res, lint_res, tmp_path)
    assert result.passed is True
    assert result.score >= 0.90
    assert len(result.rejection_reasons) == 0


def test_quality_gate_rejects_out_of_scope_files(tmp_path):
    gate = QualityGate()

    candidate = TaskCandidate(
        task_id="task_ds_01",
        track_id="track_data_science",
        project_id="proj_ds_01",
        milestone_id="ms_01",
        title="Implement Outliers",
        description="Outlier detector",
        task_type=TaskType.NEW_MODULE,
        target_files=["src/outliers.py"],
        test_files=["tests/test_outliers.py"],
        rationale="Clean implementation",
    )

    plan = ImplementationPlan(
        task_id="task_ds_01",
        summary="Suspicious modifications",
        files=[
            GeneratedFile(relative_path="src/outliers.py", content="x = 1\n", description="Module"),
            GeneratedFile(
                relative_path="unrelated_file.py", content="evil = True\n", description="Suspicious"
            ),
        ],
    )

    test_res = TestRunResult(success=True, exit_code=0, passed_count=1)
    lint_res = LintRunResult(success=True, exit_code=0)

    result = gate.evaluate(candidate, plan, test_res, lint_res, tmp_path)
    assert result.passed is False
    assert any("Out-of-Scope File Violation" in r for r in result.rejection_reasons)


def test_quality_gate_rejects_trivial_changes(tmp_path):
    gate = QualityGate(QualityGateConfig(minimum_lines_changed=20))

    candidate = TaskCandidate(
        task_id="task_ds_01",
        track_id="track_data_science",
        project_id="proj_ds_01",
        milestone_id="ms_01",
        title="Trivial task",
        description="Trivial",
        task_type=TaskType.NEW_MODULE,
        target_files=["src/trivial.py"],
        test_files=["tests/test_trivial.py"],
        rationale="Trivial",
    )

    plan = ImplementationPlan(
        task_id="task_ds_01",
        summary="Trivial lines",
        files=[
            GeneratedFile(
                relative_path="src/trivial.py", content="a = 1\n", description="One line"
            ),
            GeneratedFile(
                relative_path="tests/test_trivial.py",
                content="def test(): assert 1 == 1\n",
                description="One test",
            ),
        ],
    )

    test_res = TestRunResult(success=True, exit_code=0, passed_count=1)
    lint_res = LintRunResult(success=True, exit_code=0)

    result = gate.evaluate(candidate, plan, test_res, lint_res, tmp_path)
    assert result.passed is False
    assert any("Trivial Change Rejection" in r for r in result.rejection_reasons)
