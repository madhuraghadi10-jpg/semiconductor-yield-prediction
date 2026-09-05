import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Semiconductor Yield Predictor",
    page_icon="🔬",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")


# =========================================================
# LOAD FEATURE NAMES
# =========================================================

@st.cache_data
def load_features():
    df = pd.read_csv("data/sample_input.csv")
    return [
        column
        for column in df.columns
        if column != "Pass/Fail"
    ]


# =========================================================
# LOAD MODEL RESULTS
# =========================================================

@st.cache_data
def load_model_results():

    return pd.read_csv(
        "outputs/model_comparison.csv"
    )


# =========================================================
# LOAD FEATURE IMPORTANCE
# =========================================================

@st.cache_data
def load_feature_importance():

    return pd.read_csv(
        "outputs/feature_importance.csv"
    )


# =========================================================
# HEADER
# =========================================================

st.title("🔬 Semiconductor Yield Predictor")

st.markdown(
    """
    ### Machine Learning for Semiconductor Manufacturing

    Predict semiconductor manufacturing **PASS / FAIL**
    outcomes using sensor measurements and analyze
    machine learning model performance.
    """
)

st.divider()


# =========================================================
# LOAD FILES
# =========================================================

try:

    model = load_model()
    feature_names = load_features()
    model_results = load_model_results()
    feature_importance = load_feature_importance()

    st.success("✅ Model and project results loaded successfully!")

except FileNotFoundError as error:

    st.error(
        f"❌ Required project file not found: {error}"
    )

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🔬 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🔮 Yield Prediction",
        "📊 Model Performance",
        "🔎 Feature Importance"
    ]
)


# =========================================================
# PAGE 1 — YIELD PREDICTION
# =========================================================

if page == "🔮 Yield Prediction":

    st.header("🔮 Yield Prediction")

    st.write(
        "Upload a CSV containing the sensor measurements "
        "required by the trained model."
    )

    uploaded_file = st.file_uploader(
        "Upload sensor CSV",
        type=["csv"]
    )


    if uploaded_file is not None:

        data = pd.read_csv(
            uploaded_file
        )

        st.subheader("📋 Uploaded Data")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Rows",
                data.shape[0]
            )

        with col2:

            st.metric(
                "Columns",
                data.shape[1]
            )


        st.dataframe(
            data.head(),
            use_container_width=True
        )


        # -------------------------------------------------
        # CHECK REQUIRED FEATURES
        # -------------------------------------------------

        missing_features = [
            feature
            for feature in feature_names
            if feature not in data.columns
        ]


        if missing_features:

            st.error(
                f"❌ {len(missing_features)} sensor features "
                "are missing."
            )

            st.write(
                missing_features[:20]
            )

            st.stop()


        # -------------------------------------------------
        # PREPARE DATA
        # -------------------------------------------------

        prediction_data = data[
            feature_names
        ].copy()


        # -------------------------------------------------
        # PREDICT
        # -------------------------------------------------

        if st.button(
            "🔮 Predict Yield",
            use_container_width=True
        ):

            predictions = model.predict(
                prediction_data
            )

            probabilities = model.predict_proba(
                prediction_data
            )


            # -------------------------------------------------
            # ADD PREDICTIONS
            # -------------------------------------------------

            results = data.copy()

            results["Prediction"] = [
                "PASS" if prediction == 0
                else "FAIL"
                for prediction in predictions
            ]

            results["Pass Probability"] = (
                probabilities[:, 0]
            )

            results["Fail Probability"] = (
                probabilities[:, 1]
            )


            # -------------------------------------------------
            # SUMMARY
            # -------------------------------------------------

            pass_count = (
                predictions == 0
            ).sum()

            fail_count = (
                predictions == 1
            ).sum()


            st.divider()

            st.subheader(
                "📊 Prediction Summary"
            )


            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "Total Samples",
                    len(predictions)
                )


            with col2:

                st.metric(
                    "Predicted PASS",
                    pass_count
                )


            with col3:

                st.metric(
                    "Predicted FAIL",
                    fail_count
                )


            # -------------------------------------------------
            # RESULTS
            # -------------------------------------------------

            st.subheader(
                "🔎 Prediction Results"
            )

            st.dataframe(
                results[
                    [
                        "Prediction",
                        "Pass Probability",
                        "Fail Probability"
                    ]
                ],
                use_container_width=True
            )


            # -------------------------------------------------
            # DOWNLOAD
            # -------------------------------------------------

            csv = results.to_csv(
                index=False
            ).encode("utf-8")


            st.download_button(
                "⬇️ Download Predictions",
                csv,
                "semiconductor_predictions.csv",
                "text/csv",
                use_container_width=True
            )


# =========================================================
# PAGE 2 — MODEL PERFORMANCE
# =========================================================

elif page == "📊 Model Performance":

    st.header("📊 Model Performance")

    st.write(
        "Comparison of the machine learning models "
        "after hyperparameter tuning."
    )


    # -----------------------------------------------------
    # FIND BEST MODEL
    # -----------------------------------------------------

    best_index = model_results[
        "Test Accuracy"
    ].idxmax()

    best_model = model_results.loc[
        best_index
    ]


    # -----------------------------------------------------
    # BEST MODEL METRICS
    # -----------------------------------------------------

    st.subheader("🏆 Best Model")


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Model",
            best_model["Model"]
        )


    with col2:

        st.metric(
            "Test Accuracy",
            f"{best_model['Test Accuracy']:.2%}"
        )


    with col3:

        st.metric(
            "ROC-AUC",
            f"{best_model['ROC-AUC']:.2%}"
        )


    # -----------------------------------------------------
    # MODEL TABLE
    # -----------------------------------------------------

    st.subheader(
        "Model Comparison"
    )

    display_results = model_results[
        [
            "Model",
            "CV Accuracy",
            "Train Accuracy",
            "Test Accuracy",
            "ROC-AUC"
        ]
    ].copy()


    st.dataframe(
        display_results.style.format(
            {
                "CV Accuracy": "{:.2%}",
                "Train Accuracy": "{:.2%}",
                "Test Accuracy": "{:.2%}",
                "ROC-AUC": "{:.2%}"
            }
        ),
        use_container_width=True
    )


    # -----------------------------------------------------
    # MODEL CHART
    # -----------------------------------------------------

    st.subheader(
        "Accuracy Comparison"
    )

    chart_data = model_results[
        [
            "Model",
            "Train Accuracy",
            "Test Accuracy"
        ]
    ].set_index("Model")


    st.bar_chart(
        chart_data
    )


# =========================================================
# PAGE 3 — FEATURE IMPORTANCE
# =========================================================

elif page == "🔎 Feature Importance":

    st.header("🔎 Sensor Feature Importance")

    st.write(
        "Permutation importance shows which sensor "
        "features have the greatest effect on model "
        "performance."
    )


    # -----------------------------------------------------
    # TOP FEATURES
    # -----------------------------------------------------

    top_features = feature_importance.head(20)


    st.subheader(
        "Top 20 Important Features"
    )


    st.dataframe(
        top_features,
        use_container_width=True
    )


    # -----------------------------------------------------
    # FEATURE CHART
    # -----------------------------------------------------

    chart_data = (
        top_features
        .head(15)
        .set_index("Feature")
        .sort_values(
            "Importance"
        )
    )


    st.subheader(
        "Top 15 Features"
    )


    st.bar_chart(
        chart_data[
            "Importance"
        ]
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Semiconductor Yield Prediction | Machine Learning Project"
)