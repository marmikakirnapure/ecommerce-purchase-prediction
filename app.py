from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.joblib"
METRICS_PATH = ROOT / "outputs" / "model_comparison.csv"
SUMMARY_PATH = ROOT / "outputs" / "experiment_summary.json"

st.set_page_config(
    page_title="E-Commerce Purchase Prediction",
    page_icon="🛒",
    layout="wide"
)

@st.cache_resource
def load_bundle(path_string):
    return joblib.load(path_string)

st.title("🛒 E-Commerce Purchase Prediction")
st.caption("Machine Learning Mini-Project | Online Shoppers Purchasing Intention")

if not MODEL_PATH.exists():
    st.error("The trained model is not available yet.")
    st.markdown("Open the project folder in a terminal and run:")
    st.code("python train.py")
    st.stop()

bundle = load_bundle(str(MODEL_PATH))
model = bundle["model"]
model_name = bundle["model_name"]
features = bundle["features"]

with st.sidebar:
    st.header("About this project")
    st.write("Predicts whether a browsing session is likely to result in a purchase.")
    st.write(f"Selected model: **{model_name}**")
    st.caption(
        "This is an educational prediction tool. Its estimate is not a guarantee "
        "of a customer's behaviour."
    )

tab_predict, tab_metrics, tab_about = st.tabs(
    ["🛍️ Make a Prediction", "📊 Model Comparison", "ℹ️ About the Dataset"]
)

with tab_predict:
    st.subheader("Enter session characteristics")
    st.write(
        "Use the form to describe a session. The model was trained on session-level "
        "features; this is not a guarantee of early-session prediction."
    )

    with st.form("prediction_form"):
        st.markdown("#### Page activity")
        c1, c2, c3 = st.columns(3)
        with c1:
            administrative = st.number_input("Administrative pages", 0, 100, 0)
            administrative_duration = st.number_input(
                "Administrative duration (seconds)", min_value=0.0, value=0.0, step=10.0
            )
        with c2:
            informational = st.number_input("Informational pages", 0, 100, 0)
            informational_duration = st.number_input(
                "Informational duration (seconds)", min_value=0.0, value=0.0, step=10.0
            )
        with c3:
            product_related = st.number_input("Product-related pages", 0, 500, 1)
            product_related_duration = st.number_input(
                "Product-related duration (seconds)", min_value=0.0, value=60.0, step=30.0
            )

        st.markdown("#### Session behaviour")
        c4, c5, c6 = st.columns(3)
        with c4:
            bounce_rates = st.number_input(
                "Bounce rate", min_value=0.0, max_value=1.0, value=0.02, step=0.01
            )
            exit_rates = st.number_input(
                "Exit rate", min_value=0.0, max_value=1.0, value=0.05, step=0.01
            )
        with c5:
            special_day = st.selectbox(
                "Special-day proximity", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            )
            month = st.selectbox(
                "Month",
                ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                index=4
            )
        with c6:
            weekend = st.selectbox("Weekend session?", [False, True])
            visitor_type = st.selectbox(
                "Visitor type", ["Returning_Visitor", "New_Visitor", "Other"]
            )

        st.markdown("#### Technical and traffic context")
        c7, c8, c9 = st.columns(3)
        with c7:
            operating_system = st.selectbox("Operating system code", list(range(1, 9)), index=1)
        with c8:
            browser = st.selectbox("Browser code", list(range(1, 14)), index=1)
        with c9:
            region = st.selectbox("Region code", list(range(1, 10)), index=0)
        traffic_type = st.selectbox("Traffic type code", list(range(1, 21)), index=1)

        submitted = st.form_submit_button(
            "Predict Purchase Intention", type="primary", use_container_width=True
        )

    if submitted:
        input_data = pd.DataFrame([{
            "Administrative": administrative,
            "Administrative_Duration": administrative_duration,
            "Informational": informational,
            "Informational_Duration": informational_duration,
            "ProductRelated": product_related,
            "ProductRelated_Duration": product_related_duration,
            "BounceRates": bounce_rates,
            "ExitRates": exit_rates,
            "SpecialDay": special_day,
            "Month": month,
            "OperatingSystems": operating_system,
            "Browser": browser,
            "Region": region,
            "TrafficType": traffic_type,
            "VisitorType": visitor_type,
            "Weekend": weekend,
        }])
        input_data = input_data[features]
        prediction = int(model.predict(input_data)[0])
        probability = float(model.predict_proba(input_data)[0, 1])

        st.divider()
        st.subheader("Prediction result")
        left, right = st.columns(2)
        with left:
            if prediction == 1:
                st.success("Predicted outcome: Purchase")
            else:
                st.info("Predicted outcome: No purchase")
        with right:
            st.metric("Estimated purchase score", f"{probability:.1%}")
        st.progress(probability)
        st.caption(
            "The displayed score is a model estimate and may not be calibrated. "
            "It should not be interpreted as a guarantee."
        )
        with st.expander("View submitted features"):
            st.dataframe(input_data, use_container_width=True)

with tab_metrics:
    st.subheader("Comparison of the four tuned models")
    st.write(
        "All models use the same stratified 80:20 holdout split. Hyperparameters are "
        "selected with 3-fold stratified cross-validation on the training split only."
    )
    if METRICS_PATH.exists():
        results = pd.read_csv(METRICS_PATH)
        display_cols = [
            "Model", "CV_F1_Mean", "Accuracy", "Precision_Purchase",
            "Recall_Purchase", "F1_Purchase", "ROC_AUC",
            "PR_AUC_Average_Precision"
        ]
        st.dataframe(
            results[display_cols].round(4),
            use_container_width=True,
            hide_index=True
        )
        st.bar_chart(results.set_index("Model")[["F1_Purchase"]])
    else:
        st.warning("Run train.py to generate model-comparison results.")

    for filename, caption in [
        ("model_comparison.png", "Test-set F1-score comparison"),
        ("confusion_matrix.png", "Confusion matrix for the selected model"),
        ("roc_curve.png", "ROC curve for the selected model"),
        ("class_distribution.png", "Purchase versus non-purchase class distribution"),
        ("feature_importance.png", "Feature importance, if generated"),
    ]:
        image_path = ROOT / "outputs" / filename
        if image_path.exists():
            st.image(str(image_path), caption=caption, use_container_width=True)

with tab_about:
    st.subheader("Dataset information")
    st.markdown(
        """
        **Dataset:** Online Shoppers Purchasing Intention Dataset  
        **Source:** UCI Machine Learning Repository  
        **Target:** `Revenue` (purchase / no purchase)  
        **Problem type:** Binary classification  
        **Input data:** Session activity, timing, month, visitor type, and technical/traffic attributes.

        `PageValues` is excluded from the predictors because it can be closely associated
        with completed transactions and could produce an overly optimistic evaluation.

        **Dataset citation:** Sakar, C. & Kastro, Y. (2018). *Online Shoppers Purchasing
        Intention Dataset*. UCI Machine Learning Repository.
        https://doi.org/10.24432/C5F88Q
        """
    )
    if SUMMARY_PATH.exists():
        with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
            summary = json.load(f)
        st.json(summary)
    st.info(
        "The dataset is imbalanced. Review purchase-class precision, recall, F1-score "
        "and PR-AUC instead of relying on accuracy alone."
    )
