"""Check the app's raw-input contract and safe missing-model behavior."""

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from app import FEATURE_COLUMNS, ITEM_PROMPTS, TYPE_DESCRIPTIONS, build_input_frame
from model_loader import get_model_uri


class InputValidationTests(unittest.TestCase):
    def setUp(self):
        self.inputs = {item: 3 for item in ITEM_PROMPTS}
        self.inputs.update(age=25, gender="Female", hand="Right")

    def test_valid_row_keeps_training_schema(self):
        frame = build_input_frame(self.inputs)
        self.assertEqual(list(frame.columns), FEATURE_COLUMNS)
        self.assertEqual(len(frame), 1)

    def test_every_target_label_has_a_course_description(self):
        self.assertEqual(
            set(TYPE_DESCRIPTIONS),
            {"Moderate", "Resilient", "Overcontroller", "Undercontroller"},
        )
        self.assertTrue(all(TYPE_DESCRIPTIONS.values()))

    def test_rejects_missing_field_instead_of_imputing_it(self):
        del self.inputs["N1"]
        with self.assertRaisesRegex(ValueError, "missing fields: N1"):
            build_input_frame(self.inputs)

    def test_rejects_unexpected_field(self):
        self.inputs["target"] = "Moderate"
        with self.assertRaisesRegex(ValueError, "unexpected fields: target"):
            build_input_frame(self.inputs)

    def test_rejects_invalid_questionnaire_values(self):
        for value in (0, 6, None, 3.0, True):
            with self.subTest(value=value):
                self.inputs["N1"] = value
                with self.assertRaisesRegex(ValueError, "N1 must be"):
                    build_input_frame(self.inputs)

    def test_rejects_invalid_age(self):
        for value in (0, 12, 101, None, 25.5, True):
            with self.subTest(value=value):
                self.inputs["age"] = value
                with self.assertRaisesRegex(ValueError, "13 to 100"):
                    build_input_frame(self.inputs)

    def test_accepts_observed_age_boundaries(self):
        for age in (13, 100):
            with self.subTest(age=age):
                self.inputs["age"] = age
                self.assertEqual(build_input_frame(self.inputs)["age"].iat[0], age)

    def test_app_flags_age_immediately_and_rejects_it_on_submit(self):
        # Invalid age must be visible immediately and must not be replaced by
        # the last valid value when the user submits the form.
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        for age in (12, 101):
            with self.subTest(age=age):
                app = AppTest.from_file(str(app_path)).run(timeout=40)
                question = next(x for x in app.selectbox if x.key == "N6")
                question.set_value(4).run(timeout=40)
                app.number_input[0].set_value(age).run(timeout=40)
                self.assertEqual(app.number_input[0].value, age)
                self.assertEqual(next(x for x in app.selectbox if x.key == "N6").value, 4)
                self.assertEqual(len(app.error), 1)
                self.assertIn("13 to 100", app.error[0].value)
                app.button[0].click().run(timeout=40)
                self.assertEqual(len(app.error), 2)
                self.assertIn("13 to 100", app.error[1].value)
                self.assertEqual(len(app.success), 0)
                self.assertEqual(len(app.dataframe), 0)
                app.number_input[0].set_value(25).run(timeout=40)
                self.assertEqual(len(app.error), 0)

    def test_rejects_unknown_categories(self):
        for field, value in (("gender", "Unknown"), ("hand", "Unknown")):
            with self.subTest(field=field):
                self.inputs[field] = value
                with self.assertRaisesRegex(ValueError, f"valid {field} option"):
                    build_input_frame(self.inputs)
                self.inputs[field] = "Female" if field == "gender" else "Right"

    def test_empty_run_id_stops_before_model_loading(self):
        with self.assertRaisesRegex(ValueError, "Champion run ID is missing"):
            get_model_uri()

    def test_app_reports_missing_model_without_technical_details(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path)).run(timeout=40)
        app.button[0].click().run(timeout=40)
        self.assertEqual(len(app.error), 1)
        self.assertIn("Prediction unavailable", app.error[0].value)
        self.assertEqual(len(app.success), 0)
        self.assertEqual(len(app.exception), 0)


if __name__ == "__main__":
    unittest.main()
