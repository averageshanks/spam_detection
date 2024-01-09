import streamlit as st
from spam_classifier import SpamClassifier  # Adjust the import path as needed
from sklearn.metrics import accuracy_score, precision_score, recall_score
from email_extract import parse_email  # Assuming you have a function to parse email
from utils import load_lottie
from streamlit_lottie import st_lottie

st.set_page_config(page_title="Classifier", page_icon=":ninja:", layout="wide")

st.title("Email Classifier With Machine Learning")

def main():
    spam_classifier = SpamClassifier()

    X_train, X_test, Y_train, Y_test = spam_classifier.split_data()

    # Vectorize text
    (
        X_train_features,
        X_test_features,
        feature_extraction,
    ) = spam_classifier.vectorize_text(X_train, X_test)

    # Load trained models and common feature extraction instance
    logistic_regression_model, _ = spam_classifier.load_model_and_feature_extraction("logistic_regression_model.joblib", "common_feature_extraction.joblib")
    decision_tree_model, _ = spam_classifier.load_model_and_feature_extraction("decision_tree_model.joblib", "common_feature_extraction.joblib")
    random_forest_model, _ = spam_classifier.load_model_and_feature_extraction("random_forest_model.joblib", "common_feature_extraction.joblib")
    bagging_classifier_model, _ = spam_classifier.load_model_and_feature_extraction("bagging_classifier_model.joblib", "common_feature_extraction.joblib")
    GaussianNB_model, _ = spam_classifier.load_model_and_feature_extraction("gaussian_nb_model.joblib", "common_feature_extraction.joblib")
    svm_model, _ = spam_classifier.load_model_and_feature_extraction("svm_model.joblib", "common_feature_extraction.joblib")
    mlp_model, _ = spam_classifier.load_model_and_feature_extraction("mlp_model.joblib", "common_feature_extraction.joblib")

    with st.container():
        st.write("---")

        # Input text box
        user_input = st.text_area(
            "Enter the text message:", placeholder="Enter the text message:"
        )
        st.write("##")

        col1, col2 = st.columns([3, 1])

        with col1:
            # Predictions
            if st.button("Logistic Regression"):
                input_data_features = feature_extraction.transform([user_input])
                result = spam_classifier.predict_and_display(
                    logistic_regression_model, input_data_features, feature_extraction
                )
                st.success(f"Prediction (Logistic Regression): {result}")

            if st.button("Bagging Classifier"):
                input_data_features = feature_extraction.transform([user_input])
                result = spam_classifier.predict_and_display(
                    bagging_classifier_model, input_data_features, feature_extraction
                )
                st.success(f"Prediction (Bagging Classifier): {result}")

            if st.button("Random Forest"):
                input_data_features = feature_extraction.transform([user_input])
                result = spam_classifier.predict_and_display(
                    random_forest_model, input_data_features, feature_extraction
                )
                st.success(f"Prediction (Random Forest): {result}")

            if st.button("Decision Tree"):
                input_data_features = feature_extraction.transform([user_input])
                result = spam_classifier.predict_and_display(
                    decision_tree_model, input_data_features, feature_extraction
                )
                st.success(f"Prediction (Decision Tree): {result}")

            if st.button("Naive Bays"):
                input_data_features = feature_extraction.transform([user_input])
                result = spam_classifier.predict_and_display(
                    GaussianNB_model, input_data_features, feature_extraction
                )
                st.success(f"Prediction (Naive Bays): {result}")

            if st.button("SVM"):
                input_data_features = feature_extraction.transform([user_input])
                result = spam_classifier.predict_and_display(
                    svm_model, input_data_features, feature_extraction
                )
                st.success(f"Prediction (SVM): {result}")

            if st.button("MLP"):
                input_data_features = feature_extraction.transform([user_input])
                result = spam_classifier.predict_and_display(
                    mlp_model, input_data_features, feature_extraction
                )
                st.success(f"Prediction (MLP): {result}")

            if st.button("Show Results for All Models"):
                show_results_for_all_models(spam_classifier, user_input, feature_extraction)

        with col2:
            lottie_animation = load_lottie("./animation/classifier.json")
            st_lottie(lottie_animation, height=300, width=300, key="email")

    with st.container():
        st.write("---")

        # File uploader
        uploaded_file = st.file_uploader(
            "Upload an EML or TXT file", type=["eml", "txt"]
        )

        if uploaded_file is not None:
            # Display uploaded file
            st.success("File Uploaded Successfully!")

            # Parse email or text based on file type
            email_data = parse_email(uploaded_file)

            subject_text = (
                email_data["Subject"].decode("utf-8")
                if isinstance(email_data["Subject"], bytes)
                else email_data["Subject"]
            )
            body_text = (
                email_data["Body"].decode("utf-8")
                if isinstance(email_data["Body"], bytes)
                else email_data["Body"]
            )
            image_text = (
                email_data["Image_text"].decode("utf-8")
                if isinstance(email_data["Image_text"], bytes)
                else email_data["Image_text"]
            )
            html_text = (
                email_data["HTML_text"].decode("utf-8")
                if isinstance(email_data["HTML_text"], bytes)
                else email_data["HTML_text"]
            )

            email_text = (
                subject_text + " " + body_text + " " + image_text + " " + html_text
            )

            if st.button("Logistic Regression", key="logistic_regression"):
                input_data_features = feature_extraction.transform([email_text])
                result = spam_classifier.predict_and_display(
                    logistic_regression_model, email_text, feature_extraction
                )
                st.success(f"Prediction (Logistic Regression): {result}")

            if st.button("Bagging Classifier", key="bagging_classifier"):
                input_data_features = feature_extraction.transform([email_text])
                result = spam_classifier.predict_and_display(
                    bagging_classifier_model, email_text, feature_extraction
                )
                st.success(f"Prediction (Bagging Classifier): {result}")

            if st.button("Random Forest", key="random_forest"):
                input_data_features = feature_extraction.transform([email_text])
                result = spam_classifier.predict_and_display(
                    random_forest_model, email_text, feature_extraction
                )
                st.success(f"Prediction (Random Forest): {result}")

            if st.button("Decision Tree", key="decision_tree"):
                input_data_features = feature_extraction.transform([email_text])
                result = spam_classifier.predict_and_display(
                    decision_tree_model, email_text, feature_extraction
                )
                st.success(f"Prediction (Decision Tree): {result}")

            if st.button("Naive Bays", key="naive_bays"):
                input_data_features = feature_extraction.transform([email_text])
                result = spam_classifier.predict_and_display(
                    GaussianNB_model, email_text, feature_extraction
                )
                st.success(f"Prediction (Naive Bays): {result}")

            if st.button("SVM", key="svm"):
                input_data_features = feature_extraction.transform([email_text])
                result = spam_classifier.predict_and_display(
                    svm_model, email_text, feature_extraction
                )
                st.success(f"Prediction (SVM): {result}")

            if st.button("MLP", key="mlp"):
                input_data_features = feature_extraction.transform([email_text])
                result = spam_classifier.predict_and_display(
                    mlp_model, email_text, feature_extraction
                )
                st.success(f"Prediction (MLP): {result}")
                
            if st.button("Show Results for All Models", key="all"):
                show_results_for_all_models(spam_classifier, email_text, feature_extraction)

            #  To check in terminal
            # prediction_result = spam_classifier.predict_and_display(
            #     logistic_regression_model, email_text, feature_extraction
            # )
            # print(f"The email is predicted as: {prediction_result}")

def show_results_for_all_models(spam_classifier, input_text, feature_extraction):
    models = {
        "Logistic Regression": spam_classifier.load_model("logistic_regression_model.joblib"),
        "Bagging Classifier": spam_classifier.load_model("bagging_classifier_model.joblib"),
        "Random Forest": spam_classifier.load_model("random_forest_model.joblib"),
        "Decision Tree": spam_classifier.load_model("decision_tree_model.joblib"),
        "Naive Bays": spam_classifier.load_model("gaussian_nb_model.joblib"),
        "SVM": spam_classifier.load_model("svm_model.joblib"),
        "MLP": spam_classifier.load_model("mlp_model.joblib"),
    }

    with st.container():
        st.write("---")
        st.write("## Results for All Models")

        for model_name, model in models.items():
            input_data_features = feature_extraction.transform([input_text])
            result = spam_classifier.predict_and_display(model, input_data_features, feature_extraction)
            st.info(f"{model_name}: {result}")

if __name__ == "__main__":
    main()
