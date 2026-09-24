from load_data import load_data

def sequential_split(x, y):
    train_ratio = 0.7
    validation_ratio = 0.15
    if len(x) != len(y):
        raise ValueError("x и y должны иметь одинаковую длину")

    number_of_rows = len(x)
    train_end = int(number_of_rows * train_ratio)
    validation_end = int(number_of_rows * (train_ratio + validation_ratio))

    x_train = x.iloc[:train_end].copy()
    x_validation = x.iloc[train_end:validation_end].copy()
    x_test = x.iloc[validation_end:].copy()

    y_train = y.iloc[:train_end].copy()
    y_validation = y.iloc[train_end:validation_end].copy()
    y_test = y.iloc[validation_end:].copy()


    return(x_train, x_validation, x_test,
           y_train, y_validation, y_test)

if __name__ == "__main__":
    x, y = load_data(24)
    (x_train, x_validation, x_test, y_train, y_validation, y_test) = sequential_split(x, y)

    print("Train:", x_train.shape, y_train.shape)
    print("Validation:", x_validation.shape, y_validation.shape)
    print("Test:", x_test.shape, y_test.shape)
    print("\nTrain, %:")
    print(y_train.value_counts(normalize=True).mul(100).round(2))
    print("\nValidation, %:")
    print(y_validation.value_counts(normalize=True).mul(100).round(2))
    print("\nTest, %:")
    print(y_test.value_counts(normalize=True).mul(100).round(2))