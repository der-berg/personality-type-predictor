"""Collect raw questionnaire inputs and predict with the saved MLflow pipeline.

The app collects 22 raw features. It does not compute questionnaire scores or
perform imputation, scaling, or encoding outside the fitted pipeline.
"""

import logging
from numbers import Integral

import pandas as pd
import streamlit as st

from model_loader import load_champion_pipeline

LOGGER = logging.getLogger(__name__)

# Match the training-data feature order; target is an output, not a user input.
FEATURE_COLUMNS = [
    "N6", "N8", "N7", "N1", "N9", "N10", "N3", "N5", "E3", "N2",
    "E5", "N4", "E7", "E4", "age", "C4", "E1", "A4", "E10", "E9",
    "gender", "hand",
]

# These English item texts match the cited IPIP 50-item Big-Five markers.
# Preserve the original wording rather than introducing an unverified translation.
ITEM_PROMPTS = {
    "N1": "I get stressed out easily.",
    "N2": "I am relaxed most of the time.",
    "N3": "I worry about things.",
    "N4": "I seldom feel blue.",
    "N5": "I am easily disturbed.",
    "N6": "I get upset easily.",
    "N7": "I change my mood a lot.",
    "N8": "I have frequent mood swings.",
    "N9": "I get irritated easily.",
    "N10": "I often feel blue.",
    "E1": "I am the life of the party.",
    "E3": "I feel comfortable around people.",
    "E4": "I keep in the background.",
    "E5": "I start conversations.",
    "E7": "I talk to a lot of different people at parties.",
    "E9": "I don't mind being the center of attention.",
    "E10": "I am quiet around strangers.",
    "C4": "I make a mess of things.",
    "A4": "I sympathize with others' feelings.",
}

ANSWER_OPTIONS = {
    1: "1 — Disagree",
    2: "2 — Slightly disagree",
    3: "3 — Neutral",
    4: "4 — Slightly agree",
    5: "5 — Agree",
}
GENDER_OPTIONS = ("Female", "Male", "Other")
HAND_OPTIONS = ("Right", "Left", "Both")
MIN_AGE = 13
MAX_AGE = 100

# Short summaries adapted from the course introduction describe dataset labels,
# not individual people. They are not diagnostic interpretations.
TYPE_DESCRIPTIONS = {
    "Moderate": (
        "The course describes this as a balanced response pattern without "
        "pronounced extremes."
    ),
    "Resilient": (
        "The course describes this as a relatively emotionally steady response "
        "pattern, including under pressure."
    ),
    "Overcontroller": (
        "The course describes this as a more worried and less outgoing response "
        "pattern."
    ),
    "Undercontroller": (
        "The course describes this as a more impulsive and less rule-focused "
        "response pattern."
    ),
}


@st.cache_resource
def get_pipeline():
    """Cache the fitted pipeline so reruns do not load it from MLflow again.

    This returns an already-trained pipeline; the app does not retrain it.
    """

    return load_champion_pipeline()


def build_input_frame(inputs: dict[str, int | str]) -> pd.DataFrame:
    """Validate inputs and return one row in the training feature order.

    Missing fields must not silently become imputed values. The explicit
    ``columns`` argument preserves the order expected by the fitted pipeline.
    """

    required = set(FEATURE_COLUMNS)
    missing = required - inputs.keys()
    unexpected = inputs.keys() - required
    if missing or unexpected:
        parts = []
        if missing:
            parts.append(f"missing fields: {', '.join(sorted(missing))}")
        if unexpected:
            parts.append(f"unexpected fields: {', '.join(sorted(unexpected))}")
        raise ValueError(f"Please check the input fields ({'; '.join(parts)}).")

    for item in ITEM_PROMPTS:
        value = inputs[item]
        # bool is an int subclass in Python, but not a questionnaire answer.
        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or value not in ANSWER_OPTIONS
        ):
            raise ValueError(f"{item} must be a whole-number answer from 1 to 5.")

    age = inputs["age"]
    if (
        isinstance(age, bool)
        or not isinstance(age, Integral)
        or not MIN_AGE <= age <= MAX_AGE
    ):
        raise ValueError("Age must be a whole number from 13 to 100.")

    for field, allowed in (("gender", GENDER_OPTIONS), ("hand", HAND_OPTIONS)):
        if not isinstance(inputs[field], str) or inputs[field] not in allowed:
            raise ValueError(f"Please select a valid {field} option.")

    return pd.DataFrame([inputs], columns=FEATURE_COLUMNS)


def main() -> None:
    """Render the questionnaire and display a prediction after submission."""

    st.set_page_config(page_title="Personality Type Predictor", page_icon="🧭")
    # Static presentation only; native widgets retain keyboard interaction.
    # These Streamlit selectors are internal: recheck after framework upgrades.
    st.markdown(
        """<style>
        [data-testid='stSliderTickBar'] { display: none; }
        [data-testid='stSlider'] [data-testid='stWidgetLabel'] p {
            font-size: 1rem;
        }
        .response-anchors {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            color: var(--text-color);
            font-size: 0.875rem;
            line-height: 1.5;
            margin-top: -1rem;
            margin-bottom: 0.5rem;
        }
        </style>""",
        unsafe_allow_html=True,
    )
    st.title("Personality Type Predictor")
    st.caption("An interactive course project")
    st.markdown(
        "Answer 19 statements about yourself and add a few personal details. "
        "A machine-learning model will predict one of four personality labels. "
        "The statements are drawn from a shortened Big Five questionnaire."
    )
    st.warning(
        "For learning and demonstration only. This prediction is not a "
        "psychological or clinical diagnosis and should not be used to make "
        "decisions about a person."
    )

    st.markdown(
        "**How to use this app**\n\n"
        "1. Enter your details under **About you**.\n"
        "2. Move each slider to match what is usually true for you. "
        "There are no right or wrong answers.\n"
        "3. Select **Show result** at the end."
    )
    st.info(
        "Example answers are prefilled so you can try the app. For a result "
        "based on your answers, review every statement and change the personal "
        "details before selecting Show result."
    )

    # A regular container refreshes age feedback immediately; st.form would
    # delay it until submission. Prediction still waits for the button click.
    with st.container(border=True):
        st.header("About you")
        st.markdown(
            "This course app accepts ages 13–100, matching the ages observed "
            "in the cleaned training data. This does not guarantee accurate "
            "predictions at every age."
        )
        demographic_left, demographic_right, demographic_last = st.columns(3)

        with demographic_left:
            # Do not clamp the widget: a value such as 12 must remain visible
            # so validation can reject it, rather than reuse the last valid age.
            age = st.number_input(
                "Age in years", value=25, step=1
            )
            if not MIN_AGE <= age <= MAX_AGE:
                st.error("Please enter an age from 13 to 100 years.")
        with demographic_right:
            gender = st.selectbox("Gender", options=GENDER_OPTIONS)
        with demographic_last:
            hand = st.selectbox(
                "Writing hand", options=HAND_OPTIONS,
                help="Which hand do you usually write with?",
            )

        st.header("19 statements about you")
        st.markdown(
            "Move each slider to show how much you agree with the statement, "
            "based on what is usually true for you."
        )
        st.markdown(" · ".join(ANSWER_OPTIONS.values()))
        item_answers: dict[str, int] = {}
        for item in FEATURE_COLUMNS:
            if item in ITEM_PROMPTS:
                # Formatting changes the label only; raw answers remain ints 1-5.
                item_answers[item] = st.select_slider(
                    f"{ITEM_PROMPTS[item]} ({item})",
                    options=list(ANSWER_OPTIONS),
                    value=3,
                    format_func=ANSWER_OPTIONS.get,
                    key=item,
                )
                # Static endpoint text, never user-supplied HTML.
                st.markdown(
                    '<div class="response-anchors"><span>Disagree</span>'
                    '<span>Agree</span></div>',
                    unsafe_allow_html=True,
                )

        submitted = st.button("Show result", type="primary")

    if submitted:
        # Send only raw inputs. The saved pipeline performs all learned
        # preprocessing and classification exactly as it did in training.
        raw_inputs = {
            **item_answers,
            "age": age,
            "gender": gender,
            "hand": hand,
        }
        try:
            input_frame = build_input_frame(raw_inputs)
        except ValueError as error:
            # Explain correctable input errors before loading the model.
            st.error(str(error))
            return

        try:
            # Indicate loading/prediction, not a new training run.
            with st.spinner("Preparing your result…"):
                pipeline = get_pipeline()
                prediction = pipeline.predict(input_frame)[0]
        except Exception:
            # Keep technical details in local server logs, never in the UI.
            LOGGER.exception("Prediction failed")
            st.error(
                "Prediction unavailable. Please try again or ask the project "
                "owner to check the saved model."
            )
        else:
            with st.container(border=True):
                st.header("Your result")
                st.success(f"The model predicts: **{prediction}**")
                description = TYPE_DESCRIPTIONS.get(str(prediction))
                if description:
                    st.markdown(f"**About this label:** {description}")
                st.markdown(
                    "This is an educational estimate, not a diagnosis or a "
                    "complete description of you. Change any answer and select "
                    "**Show result** again to get a new prediction."
                )
                with st.expander("See answers used by the model (technical view)"):
                    st.dataframe(input_frame, hide_index=True, width="stretch")


if __name__ == "__main__":
    main()
