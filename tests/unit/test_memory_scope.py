import tempfile
import time
import unittest
from pathlib import Path

from src.memory.long_term import MemoryEntry
from src.memory.memory_store import SharedMemoryStore
from src.orchestrator.orchestrator import Orchestrator
from src.orchestrator.schemas import AgentResult, AgentStatus, SubTask, TaskType


class KeywordEmbedder:
    dim = 4

    def encode(self, text: str) -> list[float]:
        lowered = text.lower()
        return [
            1.0 if "landsat" in lowered else 0.0,
            1.0 if "sentinel" in lowered else 0.0,
            1.0 if "failure" in lowered or "失败" in lowered else 0.0,
            0.1,
        ]


def make_entry(entry_id: str, claim: str, *, confidence: float = 0.8, metadata=None) -> MemoryEntry:
    return MemoryEntry(
        entry_id=entry_id,
        claim=claim,
        source=f"source:{entry_id}",
        confidence=confidence,
        agent_id=entry_id,
        timestamp=time.time(),
        evidence_type="secondary",
        embedding=[],
        topic="urban heat island",
        metadata=metadata or {},
    )


class MemoryScopeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "memory.db")
        self.embedder = KeywordEmbedder()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def store(self, *, user_id="", session_id="", run_id="", include_global=True) -> SharedMemoryStore:
        return SharedMemoryStore(
            db_path=self.db_path,
            embedder=self.embedder,
            user_id=user_id,
            session_id=session_id,
            run_id=run_id,
            include_global=include_global,
        )

    def test_session_memory_does_not_cross_sessions(self):
        store_a = self.store(user_id="u1", session_id="sA", run_id="rA")
        store_a.put(make_entry(
            "task-a",
            "Landsat LST research memory for Wuhan urban heat island with validated evidence.",
        ))

        store_b = self.store(user_id="u1", session_id="sB", run_id="rB")

        self.assertEqual(store_b.query_by_similarity("Landsat LST Wuhan", min_sim=0.1), [])

    def test_global_memory_is_visible_to_sessions(self):
        global_store = self.store()
        global_store.put(make_entry(
            "global-landsat",
            "Landsat Collection 2 surface temperature is a reusable global knowledge item.",
            metadata={"scope": "global"},
        ))

        scoped_store = self.store(user_id="u1", session_id="sA")
        results = scoped_store.query_by_similarity("Landsat surface temperature", min_sim=0.1)

        self.assertEqual([entry.entry_id for entry, _ in results], ["global-landsat"])

    def test_user_memory_is_visible_across_sessions_for_same_user_only(self):
        user_store = self.store(user_id="u1", session_id="sA")
        user_store.put(make_entry(
            "user-pref",
            "Landsat preferred workflow memory for this user across future research sessions.",
            metadata={"scope": "user"},
        ))

        same_user = self.store(user_id="u1", session_id="sB")
        other_user = self.store(user_id="u2", session_id="sC")

        self.assertEqual(
            [entry.entry_id for entry, _ in same_user.query_by_similarity("Landsat workflow", min_sim=0.1)],
            ["user-pref"],
        )
        self.assertEqual(other_user.query_by_similarity("Landsat workflow", min_sim=0.1), [])

    def test_metadata_is_completed_on_write(self):
        store = self.store(user_id="u1", session_id="sA", run_id="rA")
        store.put(make_entry(
            "task-meta",
            "Landsat metadata completeness memory with evidence level and task identifiers.",
            metadata={"evidence_level": "evidence_backed", "source_tier": "academic"},
        ))

        stored = store.lt.get_entry("task-meta")

        self.assertIsNotNone(stored)
        self.assertEqual(stored.session_id, "sA")
        self.assertEqual(stored.metadata["user_id"], "u1")
        self.assertEqual(stored.metadata["session_id"], "sA")
        self.assertEqual(stored.metadata["run_id"], "rA")
        self.assertEqual(stored.metadata["task_id"], "task-meta")
        self.assertEqual(stored.metadata["scope"], "session")
        self.assertEqual(stored.metadata["evidence_level"], "evidence_backed")
        self.assertEqual(stored.metadata["source_tier"], "academic")

    def test_write_scope_overrides_spoofed_metadata(self):
        store = self.store(user_id="u1", session_id="sA", run_id="rA")
        store.put(make_entry(
            "spoofed",
            "Landsat memory with spoofed user and session metadata should be corrected.",
            metadata={
                "user_id": "attacker",
                "session_id": "other-session",
                "run_id": "other-run",
            },
        ))

        stored = store.lt.get_entry("spoofed")

        self.assertIsNotNone(stored)
        self.assertEqual(stored.metadata["user_id"], "u1")
        self.assertEqual(stored.metadata["session_id"], "sA")
        self.assertEqual(stored.metadata["run_id"], "rA")

    def test_failure_memory_can_be_retrieved_within_session(self):
        store = self.store(user_id="u1", session_id="sA")
        store.put(make_entry(
            "failed-task",
            "Failure memory: Sentinel-2 cannot directly provide thermal LST without another thermal source.",
            metadata={"status": "failed", "evidence_level": "rejected"},
        ))

        results = store.query_by_similarity("Sentinel-2 failure thermal LST", min_sim=0.1)

        self.assertEqual([entry.entry_id for entry, _ in results], ["failed-task"])

    def test_orchestrator_writes_failure_memory_with_scope_metadata(self):
        store = self.store(user_id="u1", session_id="sA", run_id="rA")
        orchestrator = Orchestrator(planner=None, agent_pool=None, memory_store=store)
        orchestrator._query = "Landsat LST research"
        orchestrator._task_map = {
            "t-fail": SubTask(
                task_id="t-fail",
                task_type=TaskType.GEO_VALIDATION,
                description="Validate whether Sentinel-2 can directly provide LST.",
            )
        }

        orchestrator._sync_failure_to_memory_store(AgentResult(
            task_id="t-fail",
            status=AgentStatus.FAILED,
            output="Sentinel-2 lacks a thermal band for direct LST retrieval.",
        ))

        results = store.query_by_similarity("Sentinel-2 failure thermal LST", min_sim=0.1)

        self.assertEqual(len(results), 1)
        entry = results[0][0]
        self.assertTrue(entry.entry_id.startswith("failure:t-fail:"))
        self.assertEqual(entry.metadata["user_id"], "u1")
        self.assertEqual(entry.metadata["session_id"], "sA")
        self.assertEqual(entry.metadata["run_id"], "rA")
        self.assertEqual(entry.metadata["task_id"], "t-fail")
        self.assertEqual(entry.metadata["evidence_level"], "rejected")
        self.assertTrue(entry.metadata["failure_memory"])


if __name__ == "__main__":
    unittest.main()
