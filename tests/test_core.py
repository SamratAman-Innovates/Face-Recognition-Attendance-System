"""Basic tests that need no webcam and no FaceNet.  Run:  python -m unittest discover tests -v"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import SETTINGS
from core import database as db
from core import attendance as att
from core.attendance import ConfirmationTracker, MarkStatus
from core.recognition import FaceRecognizer


def unit(vec):
    v = np.asarray(vec, dtype=np.float32)
    return v / np.linalg.norm(v)


class DatabaseAndAttendanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        SETTINGS.database_path = Path(self.tmp.name) / "test.db"
        db.init_db()
        db.register_student("S1", "Asha", "10-A", np.array([unit([1, 0, 0])]), "m")
        db.register_student("S2", "Ravi", "10-B", np.array([unit([0, 1, 0])]), "m")

    def tearDown(self):
        self.tmp.cleanup()

    def test_duplicate_attendance_blocked(self):
        self.assertEqual(att.mark_attendance("S1", 0.9).status, MarkStatus.MARKED)
        self.assertEqual(att.mark_attendance("S1", 0.9).status, MarkStatus.ALREADY_MARKED)
        self.assertEqual(db.count_present(att.today_str()), 1)

    def test_unknown_student(self):
        self.assertEqual(att.mark_attendance("NOPE", 0.9).status, MarkStatus.NOT_FOUND)

    def test_stats_and_filters(self):
        att.mark_attendance("S1", 0.9)
        stats = att.get_dashboard_stats()
        self.assertEqual((stats["total"], stats["present"], stats["absent"]), (2, 1, 1))
        self.assertEqual(stats["percentage"], 50.0)
        self.assertEqual(len(db.query_attendance(class_name="10-A")), 1)
        self.assertEqual(len(db.query_attendance(class_name="10-B")), 0)
        self.assertEqual(len(att.weekly_trend(7)), 7)
        sheet = db.daily_sheet(att.today_str())
        self.assertEqual(sorted(sheet["Status"]), ["Absent", "Present"])

    def test_gallery_roundtrip_and_cascade(self):
        ids, mat = db.load_gallery("m")
        self.assertEqual(sorted(ids), ["S1", "S2"])
        self.assertEqual(mat.shape, (2, 3))
        att.mark_attendance("S2", 0.9)
        db.delete_student("S2")
        self.assertEqual(db.count_students(), 1)
        self.assertEqual(db.count_present(att.today_str()), 0)

    def test_summary(self):
        att.mark_attendance("S1", 0.9)
        summary = att.student_attendance_summary(att.today_str(), att.today_str())
        row = summary[summary["Student ID"] == "S1"].iloc[0]
        self.assertEqual(row["Attendance %"], 100.0)


class RecognitionTests(unittest.TestCase):
    def setUp(self):
        ids = ["A", "A", "B"]
        embs = np.array([unit([1, 0.1, 0]), unit([1, -0.1, 0]), unit([0, 1, 0])])
        self.rec = FaceRecognizer(ids, embs, threshold=0.7, margin=0.05)

    def test_match(self):
        self.assertEqual(self.rec.identify(unit([1, 0, 0.05])).student_id, "A")

    def test_unknown(self):
        self.assertIsNone(self.rec.identify(unit([0, 0, 1])).student_id)

    def test_ambiguous_rejected(self):
        self.assertIsNone(self.rec.identify(unit([1, 1.1, 0])).student_id)


class TrackerTests(unittest.TestCase):
    def test_confirmation_streak(self):
        t = ConfirmationTracker(3)
        self.assertEqual(t.update({"A"}), [])
        self.assertEqual(t.update({"A"}), [])
        self.assertEqual(t.update({"A"}), ["A"])
        self.assertEqual(t.update({"A"}), [])      # fires only once per streak
        t.update(set())                            # streak broken
        t.update({"A"}); t.update({"A"})
        self.assertEqual(t.update({"A"}), ["A"])


if __name__ == "__main__":
    unittest.main()
