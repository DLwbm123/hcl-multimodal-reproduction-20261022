import unittest
from evidence import CoverageError, CoveredMean, Page, read_mean


class CoverageTests(unittest.TestCase):
    def read(self, rows, cursor, more, **options):
        return read_mean(Page(tuple(rows), cursor), more.__getitem__, **options)

    def test_complete_without_continuation(self):
        self.assertEqual(self.read([(2, 100)], None, {}), CoveredMean(100, 0, 0, 0))

    def test_exact_boundary_with_unique_timestamps(self):
        self.assertEqual(self.read([(24, 80)], "b", {}, unique_ages=True), CoveredMean(80, 0, 0, 0))

    def test_duplicate_boundary_timestamps_are_counted(self):
        self.assertEqual(self.read([(24, 80)], "b", {"b": Page(((24, 100), (25, 70)), None)}),
                         CoveredMean(90, 1, 1, 0))

    def test_declared_uniqueness_violation_fails(self):
        with self.assertRaisesRegex(CoverageError, "ordering"):
            self.read([(1, 80)], "b", {"b": Page(((1, 100),), None)}, unique_ages=True)

    def test_hidden_page_can_only_confirm_completeness(self):
        self.assertEqual(self.read([(1, 80)], "b", {"b": Page(((25, 70),), None)}),
                         CoveredMean(80, 1, 0, 1))

    def test_empty_page_with_continuation(self):
        self.assertEqual(self.read([], "b", {"b": Page(((1, 80),), None)}), CoveredMean(80, 1, 1, 0))

    def test_multiple_reads_stop_when_boundary_is_proven(self):
        more = {"b": Page(((2, 100),), "c"), "c": Page(((24, 120), (25, 140)), "unused")}
        self.assertEqual(self.read([(1, 80)], "b", more), CoveredMean(100, 2, 2, 0))

    def test_unsorted_page_fails(self):
        with self.assertRaisesRegex(CoverageError, "ordering"):
            self.read([(2, 100), (1, 80)], None, {})

    def test_page_boundary_must_be_sorted(self):
        with self.assertRaisesRegex(CoverageError, "ordering"):
            self.read([(2, 100)], "b", {"b": Page(((1, 80),), None)})

    def test_cursor_cycle_fails(self):
        with self.assertRaisesRegex(CoverageError, "cursor"):
            self.read([], "b", {"b": Page((), "c"), "c": Page((), "b")})

    def test_page_budget_cannot_return_partial_mean(self):
        with self.assertRaisesRegex(CoverageError, "budget"):
            self.read([(1, 80)], "b", {"b": Page(((2, 100),), "c")}, max_pages=1)

    def test_no_observations_in_window(self):
        self.assertEqual(self.read([(25, 70)], "b", {}), CoveredMean(None, 0, 0, 0))

    def test_nonfinite_observation_fails(self):
        with self.assertRaisesRegex(CoverageError, "invalid observation"):
            self.read([(1, float("nan"))], None, {})


if __name__ == "__main__":
    unittest.main()
