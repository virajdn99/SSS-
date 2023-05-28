import time

def calculate_metrics(y_test, Y_predicted):
    from sklearn import metrics
    from sklearn.metrics import classification_report, confusion_matrix

    accuracy = metrics.accuracy_score(y_test, Y_predicted)
    print("Accuracy: {:.2f}%".format(accuracy * 100))

    confusion_mat = confusion_matrix(y_test, Y_predicted)
    print("Confusion Matrix:")
    print(confusion_mat)
    print("Shape of Confusion Matrix:", confusion_mat.shape)

    print("TP\tFP\tFN\tTN\tSensitivity\tSpecificity")
    for i in range(confusion_mat.shape[0]):
        TP = round(float(confusion_mat[i, i]), 2)
        FP = round(float(confusion_mat[:, i].sum()), 2) - TP
        FN = round(float(confusion_mat[i, :].sum()), 2) - TP
        TN = round(float(confusion_mat.sum().sum()), 2) - TP - FP - FN
        sensitivity = round(TP / (TP + FN), 2)
        specificity = round(TN / (TN + FP), 2)
        print("{}\t{}\t{}\t{}\t{}\t{}".format(TP, FP, FN, TN, sensitivity, specificity))

    f_score = metrics.f1_score(y_test, Y_predicted)
    print("F1 Score: {:.2f}".format(f_score))

def neural_network(dataset, class_labels, test_size):
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier

    X = pd.read_csv(dataset)
    Y = pd.read_csv(class_labels)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

    model = MLPClassifier(hidden_layer_sizes=(100), activation='logistic', random_state=42)
    model.fit(X_train, y_train)
    Y_predicted = model.predict(X_test)

    return y_test, Y_predicted


def random_forests(dataset, class_labels, test_size):
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import metrics

    X = pd.read_csv(dataset)
    Y = pd.read_csv(class_labels)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
    model = RandomForestClassifier(n_estimators=5, criterion='entropy', random_state=42)
    model.fit(X_train, y_train)
    Y_predicted = model.predict(X_test)

    return y_test, Y_predicted

def support_vector_machines(dataset, class_labels, test_size):
    import numpy as np
    from sklearn import svm
    import pandas as pd
    from sklearn.model_selection import train_test_split

    X = pd.read_csv(dataset)
    Y = pd.read_csv(class_labels)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
    model = svm.SVC(kernel='rbf', C=2.0)
    model.fit(X_train, y_train)
    Y_predicted = model.predict(X_test)

    return y_test, Y_predicted

def main():
    dataset = "Dataset.csv"
    class_labels = "Target_Labels.csv"
    test_size = 0.3

    print("\nRunning Neural Networks...")
    start_time = time.time()
    y_test, Y_predicted = neural_network(dataset, class_labels, test_size)
    calculate_metrics(y_test, Y_predicted)
    end_time = time.time()
    print("Runtime: {:.2f} seconds".format(end_time - start_time))

    print("\nRunning Random Forests...")
    start_time = time.time()
    y_test, Y_predicted = random_forests(dataset, class_labels, test_size)
    calculate_metrics(y_test, Y_predicted)
    end_time = time.time()
    print("Runtime: {:.2f} seconds".format(end_time - start_time))

    print("\nRunning Support Vector Machines...")
    start_time = time.time()
    y_test, Y_predicted = support_vector_machines(dataset, class_labels, test_size)
    calculate_metrics(y_test, Y_predicted)
    end_time = time.time()
    print("Runtime: {:.2f} seconds".format(end_time - start_time))

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print("Total Runtime: {:.2f} seconds".format(end_time - start_time))
