import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder


def load_dataset(filename):
    data = pd.read_csv(filename)
    return data


# Task1: preprocessing
def missingData(data):
    # calculate number of missing
    missing_data = data.isnull().sum()
    # choose only columns include missing data
    missing_data = missing_data[missing_data > 0]
    return missing_data.sum()


def drop_missing(data):
    DataDropped = data.dropna()
    print("Data after dropping the missing :- ")
    print(DataDropped)


def replace_missing(data):
    DataMiss = data.drop(columns=["Rain"])
    # fill the missing with mean of each feature
    FilledData = DataMiss.fillna(DataMiss.mean())
    FilledData["Rain"] = data["Rain"]
    print("Data after filling the missing :- ")
    print(FilledData)


def encode_target(data):
    enc = OrdinalEncoder(categories=[['rain', 'no rain']])
    target_encoded = enc.fit_transform(data['Rain'].values.reshape(-1, 1))
    return target_encoded


def scale_data(x_train, x_test, y_train, y_test):
    numeric_data_x_train = x_train.select_dtypes(include=['number'])
    numeric_data_x_test = x_test.select_dtypes(include=['number'])

    numeric_data_y_train = y_train.reshape(-1, 1)
    numeric_data_y_test = y_test.reshape(-1, 1)

    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(numeric_data_x_train)
    x_test_scaled = scaler.transform(numeric_data_x_test)

    y_train_scaled = scaler.fit_transform(numeric_data_y_train)
    y_test_scaled = scaler.transform(numeric_data_y_test)

    return (
        x_train_scaled,
        x_test_scaled,
        y_train_scaled,
        y_test_scaled
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
    target = encode_target(data)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)
    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


def main():
    # a) Load the dataset
    data = load_dataset("weather_forecast_data.csv")
    print(f'# of missing data : {missingData(data)}')
    print("""
    choose technique to handle data:
     1- Dropping rows of missing data
     2- Replacing with the average of the feature
    """)
    choice = input("Enter number of your choice: ")
    if choice == 1:
        drop_missing(data)
    else:
        replace_missing(data)

    x_train, x_test, y_train, y_test = split_data(data)

    if not checkScalability(data):
        x_train, x_test, y_train, y_test = scale_data(x_train, x_test, y_train, y_test)

    print(x_train)



if __name__ == "__main__":
    main()


