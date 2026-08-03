"""
Testes estruturais de .github/workflows/build_historical_dataset.yml.

Valida a configuração do workflow (parsing YAML puro, sem correr o
GitHub Actions) — sem chamadas de rede, sem BSD API: apenas garante que o
ficheiro tem os triggers, inputs e passos exigidos (workflow_dispatch
apenas, sem schedule/push; inputs competition_id/season_id/output_format/
page_size/resume; upload de artefactos com if-no-files-found: ignore).

Nota: PyYAML interpreta a chave `on:` dos workflows do GitHub Actions como
o booleano `True` (regra do YAML 1.1) — isto não afeta o GitHub Actions em
si (que trata `on:` de forma especial), só a leitura com `yaml.safe_load`
aqui nos testes; por isso aceitamos tanto `"on"` como `True` como chave.
"""

import unittest
from pathlib import Path

import yaml

WORKFLOW_PATH = (
    Path(__file__).resolve().parents[2] / ".github" / "workflows" / "build_historical_dataset.yml"
)


class TestBuildHistoricalDatasetWorkflow(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.raw_text = WORKFLOW_PATH.read_text(encoding="utf-8")
        cls.workflow = yaml.safe_load(cls.raw_text)
        cls.on_section = cls.workflow.get("on", cls.workflow.get(True))

    def test_workflow_file_exists(self):
        self.assertTrue(WORKFLOW_PATH.exists())

    def test_only_trigger_is_workflow_dispatch(self):
        self.assertEqual(list(self.on_section.keys()), ["workflow_dispatch"])

    def test_no_schedule_trigger(self):
        self.assertNotIn("schedule", self.on_section)

    def test_no_push_or_pull_request_trigger(self):
        self.assertNotIn("push", self.on_section)
        self.assertNotIn("pull_request", self.on_section)

    def test_expected_inputs_present(self):
        inputs = self.on_section["workflow_dispatch"]["inputs"]
        self.assertEqual(
            set(inputs.keys()),
            {"competition_id", "season_id", "output_format", "page_size", "resume"},
        )

    def test_output_format_is_choice_with_expected_options(self):
        output_format = self.on_section["workflow_dispatch"]["inputs"]["output_format"]
        self.assertEqual(output_format["type"], "choice")
        self.assertEqual(set(output_format["options"]), {"csv", "sqlite", "parquet", "all"})
        self.assertEqual(output_format["default"], "all")

    def test_resume_is_choice_true_false(self):
        resume = self.on_section["workflow_dispatch"]["inputs"]["resume"]
        self.assertEqual(resume["type"], "choice")
        self.assertEqual(set(resume["options"]), {"true", "false"})
        self.assertEqual(resume["default"], "false")

    def test_competition_id_and_season_id_are_optional_strings(self):
        inputs = self.on_section["workflow_dispatch"]["inputs"]
        for name in ("competition_id", "season_id"):
            self.assertEqual(inputs[name]["type"], "string")
            self.assertFalse(inputs[name]["required"])

    def test_single_job_present(self):
        self.assertEqual(len(self.workflow["jobs"]), 1)

    def _job(self):
        return next(iter(self.workflow["jobs"].values()))

    def test_job_runs_build_historical_dataset_script(self):
        job = self._job()
        run_steps = [s.get("run", "") for s in job["steps"] if "run" in s]
        self.assertTrue(any("build_historical_dataset.py" in run for run in run_steps))

    def test_build_step_passes_competition_id_season_id_output_page_size_resume(self):
        job = self._job()
        run_steps = [s.get("run", "") for s in job["steps"] if "build_historical_dataset.py" in s.get("run", "")]
        self.assertEqual(len(run_steps), 1)
        script = run_steps[0]
        for flag in ("--competition-id", "--season-id", "--output", "--page-size", "--resume"):
            self.assertIn(flag, script)

    def test_build_step_does_not_interpolate_inputs_directly_in_run(self):
        """Inputs devem chegar via env (INPUT_*), nunca `${{ inputs.* }}` interpolado dentro do bloco run:."""
        job = self._job()
        build_step = next(s for s in job["steps"] if "build_historical_dataset.py" in s.get("run", ""))
        self.assertNotIn("${{ inputs.", build_step["run"])
        env = build_step.get("env", {})
        self.assertIn("INPUT_COMPETITION_ID", env)
        self.assertIn("INPUT_SEASON_ID", env)
        self.assertIn("INPUT_OUTPUT_FORMAT", env)
        self.assertIn("INPUT_PAGE_SIZE", env)
        self.assertIn("INPUT_RESUME", env)

    def test_build_step_never_echoes_api_key_or_token(self):
        job = self._job()
        build_step = next(s for s in job["steps"] if "build_historical_dataset.py" in s.get("run", ""))
        script_lower = build_step["run"].lower()
        for forbidden in ("echo $bsd_api_key", "echo $api_key", "print(", "cat $bsd_api_key"):
            self.assertNotIn(forbidden, script_lower)

    def test_upload_artifact_step_present_with_expected_paths_and_ignore_missing(self):
        job = self._job()
        upload_steps = [s for s in job["steps"] if "upload-artifact" in s.get("uses", "")]
        self.assertEqual(len(upload_steps), 1)
        upload = upload_steps[0]

        self.assertEqual(upload["with"]["if-no-files-found"], "ignore")
        paths = upload["with"]["path"]
        for expected in (
            "data/historical/historical.csv",
            "data/historical/historical.sqlite",
            "data/historical/historical.parquet",
            "data/historical/dataset_report.json",
        ):
            self.assertIn(expected, paths)

    def test_no_commit_or_git_push_step(self):
        job = self._job()
        for step in job["steps"]:
            run = step.get("run", "")
            self.assertNotIn("git commit", run)
            self.assertNotIn("git push", run)

    def test_no_write_permissions_requested(self):
        permissions = self.workflow.get("permissions", {})
        self.assertNotEqual(permissions.get("contents"), "write")


if __name__ == "__main__":
    unittest.main()
