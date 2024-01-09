import joblib

# Load the SpamClassifier class
from spam_classifier import SpamClassifier  # Replace 'your_module' with the actual module name

if __name__ == "__main__":
    # Create an instance of the SpamClassifier class
    spam_classifier = SpamClassifier()

    # Load the common feature extraction instance
    common_feature_extraction = spam_classifier.load_feature_extraction("common_feature_extraction.joblib")

    # Load each trained model
    logistic_regression_model = spam_classifier.load_model("logistic_regression_model.joblib")
    decision_tree_model = spam_classifier.load_model("decision_tree_model.joblib")
    random_forest_model = spam_classifier.load_model("random_forest_model.joblib")
    bagging_classifier_model = spam_classifier.load_model("bagging_classifier_model.joblib")
    gaussian_nb_model = spam_classifier.load_model("gaussian_nb_model.joblib")
    svm_model = spam_classifier.load_model("svm_model.joblib")
    mlp_model = spam_classifier.load_model("mlp_model.joblib")

    # Assuming you have some test data
    test_data = ["Your test email goes here."]

    # Vectorize the test data using the common feature extraction instance
    test_data_features = common_feature_extraction.transform(test_data)

    # Make predictions using each model
    logistic_regression_prediction = spam_classifier.predict_and_display(logistic_regression_model, test_data_features, common_feature_extraction)
    decision_tree_prediction = spam_classifier.predict_and_display(decision_tree_model, test_data_features, common_feature_extraction)
    random_forest_prediction = spam_classifier.predict_and_display(random_forest_model, test_data_features, common_feature_extraction)
    bagging_classifier_prediction = spam_classifier.predict_and_display(bagging_classifier_model, test_data_features, common_feature_extraction)
    gaussian_nb_prediction = spam_classifier.predict_and_display(gaussian_nb_model, test_data_features, common_feature_extraction)
    svm_prediction = spam_classifier.predict_and_display(svm_model, test_data_features, common_feature_extraction)
    mlp_prediction = spam_classifier.predict_and_display(mlp_model, test_data_features, common_feature_extraction)

    # Print or use the predictions as needed
    print("Logistic Regression Prediction:", logistic_regression_prediction)
    print("Decision Tree Prediction:", decision_tree_prediction)
    print("Random Forest Prediction:", random_forest_prediction)
    print("Bagging Classifier Prediction:", bagging_classifier_prediction)
    print("Gaussian NB Prediction:", gaussian_nb_prediction)
    print("SVM Prediction:", svm_prediction)
    print("MLP Prediction:", mlp_prediction)
