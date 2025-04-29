import csv
import sys
from sklearn.neighbors import KNeighborsClassifier

# Dictionary to convert month strings to numbers
MONTHS = {
    "Jan": 0,
    "Feb": 1,
    "Mar": 2,
    "Apr": 3,
    "May": 4,
    "June": 5,
    "Jul": 6,
    "Aug": 7,
    "Sep": 8,
    "Oct": 9,
    "Nov": 10,
    "Dec": 11
}

def load_data(filename):
    """
    Load shopping data from a CSV file and return (evidence, labels).
    """
    evidence = []
    labels = []

    with open(filename, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            data = [
                int(row["Administrative"]),
                float(row["Administrative_Duration"]),
                int(row["Informational"]),
                float(row["Informational_Duration"]),
                int(row["ProductRelated"]),
                float(row["ProductRelated_Duration"]),
                float(row["BounceRates"]),
                float(row["ExitRates"]),
                float(row["PageValues"]),
                float(row["SpecialDay"]),
                MONTHS[row["Month"]],
                int(row["OperatingSystems"]),
                int(row["Browser"]),
                int(row["Region"]),
                int(row["TrafficType"]),
                1 if row["VisitorType"] == "Returning_Visitor" else 0,
                1 if row["Weekend"] == "TRUE" else 0
            ]
            evidence.append(data)
            labels.append(1 if row["Revenue"] == "TRUE" else 0)

    return (evidence, labels)

def train_model(evidence, labels):
    """
    Train a k-nearest neighbor classifier (k=1) on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model

def evaluate(labels, predictions):
    """
    Evaluate the model performance:
    Return (sensitivity, specificity).
    """
    true_positives = sum(1 for true, pred in zip(labels, predictions) if true == 1 and pred == 1)
    false_negatives = sum(1 for true, pred in zip(labels, predictions) if true == 1 and pred == 0)
    true_negatives = sum(1 for true, pred in zip(labels, predictions) if true == 0 and pred == 0)
    false_positives = sum(1 for true, pred in zip(labels, predictions) if true == 0 and pred == 1)

    sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) != 0 else 0

    return (sensitivity, specificity)

def main():
    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    filename = sys.argv[1]

    # Load data
    evidence, labels = load_data(filename)

    # Train model
    model = train_model(evidence, labels)

    # Make predictions
    predictions = model.predict(evidence)

    # Evaluate
    sensitivity, specificity = evaluate(labels, predictions)

    # Print results
    print(f"Correct: {sum(1 for true, pred in zip(labels, predictions) if true == pred)}")
    print(f"Incorrect: {sum(1 for true, pred in zip(labels, predictions) if true != pred)}")
    print(f"True Positive Rate: {sensitivity:.2f}")
    print(f"True Negative Rate: {specificity:.2f}")

if __name__ == "__main__":
    main()
