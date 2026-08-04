"""
Testes do agendamento (`schedule`) do workflow `.github/workflows/live_logger.yml`.

Cobrem:
  - o `schedule` novo (execução de 2 em 2 minutos, entre as 08:00 e as
    23:58 UTC, todos os dias) foi adicionado;
  - `workflow_dispatch` continua disponível (nenhuma funcionalidade
    existente foi removida);
  - a execução agendada (`schedule`) corre exclusivamente o Scanner
    (`src/alerts/live_scanner.py` — apenas Alerta Live Premium);
  - a execução manual (`workflow_dispatch`) continua a correr o pipeline
    completo já existente (`src/engine/live_monitor.py`), inalterado;
  - o estado de anti-spam (`data/live_alerts.db`) é persistido no commit
    automático, para sobreviver entre execuções agendadas em runners
    efémeros do GitHub Actions.

Nota: o PyYAML (YAML 1.1) interpreta a chave "on:" como o booleano `True`
— por isso o teste procura os triggers em `workflow.get("on", workflow[True])`.
"""

import unittest
from pathlib import Path

import yaml

WORKFLOW_PATH = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "live_logger.yml"


class TestLiveLoggerWorkflowSchedule(unittest.TestCase):

    def setUp(self):
        with open(WORKFLOW_PATH, encoding="utf-8") as f:
            self.workflow = yaml.safe_load(f)
        self.triggers = self.workflow.get("on", self.workflow.get(True))
        self.steps = self.workflow["jobs"]["run-logger"]["steps"]

    def test_triggers_were_parsed(self):
        self.assertIsNotNone(self.triggers)

    def test_has_schedule_trigger_every_two_minutes_between_08_and_23_58_utc(self):
        self.assertIn("schedule", self.triggers)
        crons = [entry["cron"] for entry in self.triggers["schedule"]]
        self.assertIn("*/2 8-23 * * *", crons)

    def test_still_has_workflow_dispatch(self):
        self.assertIn("workflow_dispatch", self.triggers)

    def test_scheduled_run_executes_live_scanner_exclusively(self):
        scanner_steps = [s for s in self.steps if "src/alerts/live_scanner.py" in s.get("run", "")]
        self.assertEqual(len(scanner_steps), 1)
        self.assertEqual(scanner_steps[0].get("if"), "github.event_name == 'schedule'")

    def test_manual_dispatch_still_executes_full_live_monitor_pipeline(self):
        monitor_steps = [s for s in self.steps if "src/engine/live_monitor.py" in s.get("run", "")]
        self.assertEqual(len(monitor_steps), 1)
        self.assertEqual(monitor_steps[0].get("if"), "github.event_name != 'schedule'")

    def test_existing_label_recalculation_step_still_present(self):
        label_steps = [s for s in self.steps if "create_labels.py" in s.get("run", "")]
        self.assertEqual(len(label_steps), 1)

    def test_persists_live_alerts_db_for_anti_spam_state(self):
        commit_step = next(s for s in self.steps if "Guardar Alterações" in s["name"])
        self.assertIn("data/live_alerts.db", commit_step["run"])
        self.assertIn("data/live_history.db", commit_step["run"])

    def test_concurrency_group_unchanged(self):
        self.assertEqual(self.workflow["concurrency"]["group"], "live-logger")


if __name__ == "__main__":
    unittest.main()
