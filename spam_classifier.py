import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Bagging Classifier**
from sklearn.ensemble import BaggingClassifier

# Naive Bayes**
from sklearn.naive_bayes import GaussianNB

# Support Vector Machine
from sklearn.svm import SVC

# Multi-layer Preceptron Neural Network
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib


class SpamClassifier:
    def __init__(self, file_path="mail_data.csv"):
        self.df = pd.read_csv(file_path)
        self.data = self.clean_data()

    def clean_data(self):
        data = self.df.dropna()
        data.loc[
            data["Category"] == "spam", "Category"
        ] = 1  # Change 'spam' to 1 (positive class)
        data.loc[
            data["Category"] == "ham", "Category"
        ] = 0  # Change 'ham' to 0 (negative class)
        data["Category"] = data["Category"].astype(int)  # Convert to integer type
        return data

    def split_data(self, test_size=0.2, random_state=3):
        X = self.data["Message"]
        Y = self.data["Category"]
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, Y_train, Y_test

    def vectorize_text(
        self, X_train, X_test, min_df=1, stop_words="english", lowercase=True
    ):
        feature_extraction = TfidfVectorizer(
            min_df=min_df, stop_words=stop_words, lowercase=lowercase
        )
        X_train_features = feature_extraction.fit_transform(X_train)
        X_test_features = feature_extraction.transform(X_test)
        return X_train_features, X_test_features, feature_extraction

    def train_logistic_regression(self, X_train_features, Y_train):
        model = LogisticRegression()
        model.fit(X_train_features, Y_train)
        return model

    def train_decision_tree(self, X_train_features, Y_train, random_state=1):
        model = DecisionTreeClassifier(random_state=random_state)
        model.fit(X_train_features, Y_train)
        return model

    def train_random_forest(self, X_train_features, Y_train, random_state=1):
        model = RandomForestClassifier(random_state=random_state)
        model.fit(X_train_features, Y_train)
        return model

    def train_bagging_classifier(self, X_train_features, Y_train, random_state=1):
        model = BaggingClassifier(random_state=random_state)
        model.fit(X_train_features, Y_train)
        return model

    def train_GaussianNB(self, X_train_features, Y_train):
        model = GaussianNB()
        X_train_array = X_train_features.toarray()
        model.fit(X_train_array, Y_train)
        return model

    def train_svm(self, X_train_features, Y_train, random_state=1):
        model = SVC(kernel="linear", random_state=42)
        model.fit(X_train_features, Y_train)
        return model
    
    def train_mlp(self, X_train_feature, Y_train):
        model = MLPClassifier()
        model.fit(X_train_feature, Y_train)
        return model

    def calculate_metrics(self, model, X_test_features, Y_test):
        prediction = model.predict(X_test_features)
        accuracy = accuracy_score(Y_test, prediction)
        precision = precision_score(Y_test, prediction)
        recall = recall_score(Y_test, prediction)
        return accuracy, precision, recall

    def predict_and_display(self, model, input_data_features, feature_extraction):
        if isinstance(input_data_features, str):
            # If the input data is a string, assume it's a single text sample
            input_data_features = feature_extraction.transform([input_data_features])

        input_data_array = input_data_features.toarray()
        prediction = model.predict(input_data_array)
        result = (
            "ham" if prediction[0] == 0 else "spam"
        )  # Reverse the encoding for display
        return result
    
    def save_model(self, model, filename):
        joblib.dump(model, filename)

    def load_model(self, filename):
        return joblib.load(filename)

    def save_feature_extraction(self, feature_extraction, filename):
        joblib.dump(feature_extraction, filename)

    def load_feature_extraction(self, filename):
        return joblib.load(filename)

    def save_model_and_feature_extraction(self, model, feature_extraction, model_filename, feature_extraction_filename):
        self.save_model(model, model_filename)
        self.save_feature_extraction(feature_extraction, feature_extraction_filename)

    def load_model_and_feature_extraction(self, model_filename, feature_extraction_filename):
        model = self.load_model(model_filename)
        feature_extraction = self.load_feature_extraction(feature_extraction_filename)
        return model, feature_extraction


    
if __name__ == "__main__":
    spam_classifier = SpamClassifier()

    # Assuming you have trained a model (e.g., logistic regression)
    X_train, X_test, Y_train, Y_test = spam_classifier.split_data()
    
    # Use a single feature extraction instance for all models
    X_train_features, X_test_features, feature_extraction = spam_classifier.vectorize_text(X_train, X_test)

    logistic_regression_model = spam_classifier.train_logistic_regression(X_train_features, Y_train)
    decision_tree_model = spam_classifier.train_decision_tree(X_train_features, Y_train)
    random_forest_model = spam_classifier.train_random_forest(X_train_features, Y_train)
    bagging_classifier_model = spam_classifier.train_bagging_classifier(X_train_features, Y_train)
    gaussian_nb_model = spam_classifier.train_GaussianNB(X_train_features, Y_train)
    svm_model = spam_classifier.train_svm(X_train_features, Y_train)
    mlp_model = spam_classifier.train_mlp(X_train_features, Y_train)

    # Save the trained model using save_model function
    spam_classifier.save_model(logistic_regression_model, "logistic_regression_model.joblib")
    spam_classifier.save_model(decision_tree_model, "decision_tree_model.joblib")
    spam_classifier.save_model(random_forest_model, "random_forest_model.joblib")
    spam_classifier.save_model(bagging_classifier_model, "bagging_classifier_model.joblib")
    spam_classifier.save_model(gaussian_nb_model, "gaussian_nb_model.joblib")
    spam_classifier.save_model(svm_model, "svm_model.joblib")
    spam_classifier.save_model(mlp_model, "mlp_model.joblib")

    # Save the common feature extraction instance
    spam_classifier.save_feature_extraction(feature_extraction, "common_feature_extraction.joblib")
