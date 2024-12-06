import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


def load_dataset(filename):
    data = pd.read_csv(filename)
    return data


# Task1: preprocessing
def missingData(data):
    missing_data = data.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    return missing_data.sum()


def drop_missing(data):
    rows_before = data.shape[0]
    DataDropped = data.dropna()
    rows_after = DataDropped.shape[0]

    print(f"Number of rows before dropping: {rows_before}")
    print(f"Number of rows after dropping: {rows_after}")
    print(f"Number of rows dropped: {rows_before - rows_after}")

    return DataDropped


def replace_missing(data):
    DataMiss = data.drop(columns=["Rain"])
    # fill the missing with mean of each feature
    FilledData = DataMiss.fillna(DataMiss.mean())
    FilledData["Rain"] = data["Rain"]
    return FilledData


def scale_data(x_train, x_test):
    numeric_data_x_train = x_train.select_dtypes(include=['number'])
    numeric_data_x_test = x_test.select_dtypes(include=['number'])

    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(numeric_data_x_train)
    x_test_scaled = scaler.transform(numeric_data_x_test)

    return (
        x_train_scaled,
        x_test_scaled,

    )


def checkScalability(data):
    numeric_data = data.select_dtypes(include=['number'])
    data_std = numeric_data.std()
    data_std = data_std.to_dict()
    is_scalable = True
    for col, std_value in data_std.items():
        if not (0.9 <= std_value <= 1.1):
            is_scalable = False
            break

    if is_scalable:
        print("\nData is already standardized with std ≈ 1 for all features.")
    else:
        print("\nScaling is required for the data.")
    return is_scalable


def split_data(data):
    features = data.drop(columns=['Rain'])
    target = data['Rain']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)
    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# Calculate the Euclidean distance between two points
def euclidean_distance(point1, point2):
    difference = point1 - point2
    squared_difference = np.power(difference, 2)
    distance = np.sqrt(np.sum(squared_difference))
    return distance


def knn_from_scratch(X_train, y_train, X_test, k):
    y_predicted = []
    for test_point in X_test:
        distances = []
        for i in range(len(X_train)):
            train_point = X_train[i]
            distances.append((euclidean_distance(test_point, train_point), y_train.iloc[i]))
        distances.sort(key=lambda x: x[0])
        k_elements = distances[:k]

        # Extract the values
        neighbors_values = [value for distance, value in k_elements]

        # Count occurrences manually
        frequencies = {}
        for value in neighbors_values:
            frequencies[value] = frequencies.get(value, 0) + 1

        # Find the value with the maximum count
        predection = max(frequencies, key=frequencies.get)
        y_predicted.append(predection)

    return y_predicted


def knn_comparison(x_train, y_train, x_test, y_test):
    # Prepare data for the comparison table
    comparison_data = {
        "Model": [],
        "Accuracy": [],
        "Precision": [],
        "Recall": []
    }

    # KNN from Scratch Evaluation for Different k Values
    k_values = [3, 5, 7, 9, 11]
    for k in k_values:
        # KNN Using scikit-learn
        knn_model = KNeighborsClassifier(n_neighbors=k)
        knn_model.fit(x_train, y_train)
        y_pred_knn = knn_model.predict(x_test)

        # Evaluate scikit-learn kNN
        accuracy_knn = accuracy_score(y_test, y_pred_knn)
        precision_knn = precision_score(y_test, y_pred_knn, pos_label='rain')
        recall_knn = recall_score(y_test, y_pred_knn, pos_label='rain')

        # Add scikit learn kNN metrics to the comparison data
        comparison_data["Model"].append(f"Scikit-learn kNN (k={k})")
        comparison_data["Accuracy"].append(accuracy_knn)
        comparison_data["Precision"].append(precision_knn)
        comparison_data["Recall"].append(recall_knn)

        y_pred_custom_knn = knn_from_scratch(x_train, y_train, x_test, k)
        accuracy_custom_knn = accuracy_score(y_test, y_pred_custom_knn)
        precision_custom_knn = precision_score(y_test, y_pred_custom_knn, pos_label='rain')
        recall_custom_knn = recall_score(y_test, y_pred_custom_knn, pos_label='rain')

        # Add custom kNN metrics to the comparison data
        comparison_data["Model"].append(f"Custom kNN (k={k})")
        comparison_data["Accuracy"].append(accuracy_custom_knn)
        comparison_data["Precision"].append(precision_custom_knn)
        comparison_data["Recall"].append(recall_custom_knn)

    # Convert to DataFrame for better formatting
    df_comparison = pd.DataFrame(comparison_data)
    return df_comparison


def algorithms(x_train, y_train, x_test, y_test):
    algorithms_implementation(x_train, y_train, x_test, y_test, "decision tree")
    algorithms_implementation(x_train, y_train, x_test, y_test, "KNN")
    algorithms_implementation(x_train, y_train, x_test, y_test, "Naïve Bayes")


def algorithms_implementation(x_train, y_train, x_test, y_test, algorithm):
    if algorithm == "decision tree":
        model = DecisionTreeClassifier(random_state=42)
    elif algorithm == "KNN":
        model = KNeighborsClassifier(n_neighbors=3)
    else:
        model = GaussianNB()

    model.fit(x_train, y_train)
    prediction_values = model.predict(x_test)
    performance_comparison(prediction_values,y_test,algorithm)


def performance_comparison(prediction_values, y_test, algorithm):
    metrics_comparison = {
        "Model": [],
        "Accuracy": [],
        "Precision": [],
        "Recall": []
    }

    # evaluation
    accuracy = accuracy_score(y_test, prediction_values)
    precision = precision_score(y_test, prediction_values, pos_label='rain')
    recall = recall_score(y_test, prediction_values, pos_label='rain')
    # add performance to metrics
    metrics_comparison["Model"].append(algorithm)
    metrics_comparison["Accuracy"].append(accuracy)
    metrics_comparison["Precision"].append(precision)
    metrics_comparison["Recall"].append(recall)
    # print report
    comparison_table = pd.DataFrame(metrics_comparison)
    print("-----------------------------")
    print("Performance Comparison Table:")
    print(comparison_table)


def report_handling_data(data):
    data_after_drop = drop_missing(data)
    x_train_d, x_test_d, y_train_d, y_test_d = split_data(data_after_drop)
    print("-----------------------------")
    print("data after dropping technique")
    algorithms(x_train_d, y_train_d, x_test_d, y_test_d)

    # ---------------------------------------------------------------

    data_after_replace = replace_missing(data)
    x_train_r, x_test_r, y_train_r, y_test_r = split_data(data_after_replace)
    print("-----------------------------")
    print("data after replacing technique")
    algorithms(x_train_r, y_train_r, x_test_r, y_test_r)


def decision_tree_plot(x_train, y_train, columns):
    # convert x_train back to a DataFrame to retain column names
    x_train = pd.DataFrame(x_train, columns=columns)

    # plot of the decision tree
    model = DecisionTreeClassifier(random_state=42)
    model.fit(x_train, y_train)

    plt.figure(figsize=(12, 12))
    plot_tree(
        model,
        feature_names=x_train.columns,
        class_names=model.classes_,
        filled=True,
        rounded=True
    )
    plt.title("Decision Tree Structure")
    plt.savefig("decision_tree_plot.png")
    plt.show()

    # detailed report how to get prediction
    print("\nFeatures Importance:")
    features_importance = model.feature_importances_
    importance_data = pd.DataFrame({
        'Feature': columns,
        'Importance': features_importance
    })
    importance_data = importance_data.sort_values(by='Importance', ascending=False)

    for feature, importance in importance_data['Importance'].items():
        print(f"Feature '{feature}' has importance: {importance}")
    # -----------------------------------------------------------------------

    # Splitting Logic
    print("\nSplitting Logic:")
    tree_ = model.tree_
    for i in range(tree_.node_count):
        if tree_.feature[i] != -2:
            feature = columns[tree_.feature[i]]
            decision_boundry = tree_.threshold[i]
            print(f"""
            Node {i}: Splits on '{feature}' <= {decision_boundry}
            Left child (Node {tree_.children_left[i]}): '{feature} <= {decision_boundry}
            Right child (Node {tree_.children_right[i]}): '{feature} > {decision_boundry}\n""")


def main():
    # a) Load the dataset
    data = load_dataset("weather_forecast_data.csv")
    data_before_preprocess = data  # for affect handling

    print(f'# of missing data : {missingData(data)}')
    print("""
    choose technique to handle data:
     1- Dropping rows of missing data
     2- Replacing with the average of the feature
    """)
    choice = input("Enter number of your choice: ")
    if choice == '1':
        data = drop_missing(data)
    else:
        data = replace_missing(data)

    x_train, x_test, y_train, y_test = split_data(data)
    columns = x_train.columns
    if not checkScalability(data):
        x_train, x_test = scale_data(x_train, x_test)

    # Apply knn algorithm and Display the comparison table
    df_comparison = knn_comparison(x_train, y_train, x_test, y_test)
    print("\nKNN Performance Comparison (scikit-learn vs Custom kNN):")
    print(df_comparison)
    algorithms(x_train, y_train, x_test, y_test)
    report_handling_data(data_before_preprocess)
    decision_tree_plot(x_train, y_train, columns)


if __name__ == "__main__":
    main()
