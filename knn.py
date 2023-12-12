import pandas as pd
import random

# Function to split data into training and testing sets
def SplitData(data, ratio):
    data_length = len(data)
    split_point = int(data_length * ratio)
    random.shuffle(data)
    train_data = data[:split_point]
    test_data = data[split_point:]
    return train_data, test_data

# Function to perform Min-Max Scaling
def MinMaxScaling(data):
    num_features = len(data[0]) - 1
    for i in range(num_features):
        col_values = [row[i] for row in data]
        min_val = min(col_values)
        max_val = max(col_values)
        for row in data:
            row[i] = (row[i] - min_val) / (max_val - min_val)
    return data

# Function to calculate Euclidean distance between two points
def EuclideanDistance(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

# Function for KNN classification with distance-weighted voting
def KnnClassification(train_data, test_instance, k):
    d_list = []
    for train_instance in train_data:
        d = EuclideanDistance(test_instance[:-1], train_instance[:-1])
        d_list.append((train_instance, d))

    d_list.sort(key=lambda x: x[1])

    neighbors = d_list[:k]

    if len(set(neighbor[1] for neighbor in neighbors)) == 1:
        # There is a tie, implement distance-weighted voting
        weighted_votes = {}
        for neighbor, distance in neighbors:
            label = neighbor[-1]
            if distance == 0:
                return label  # Avoid division by zero
            weight = 1 / distance
            weighted_votes[label] = weighted_votes.get(label, 0) + weight

        # Return the label with the highest weighted vote
        return max(weighted_votes, key=weighted_votes.get)

    # No tie, return the most frequent class
    neighbor_list = [neighbor[0][-1] for neighbor in neighbors]
    return max(set(neighbor_list), key=neighbor_list.count)

# Function to test the KNN classifier
def TestKnn(train_data, test_data, k):
    right_predictions = 0
    datatest_length = len(test_data)

    for test_a in test_data:
        predicted = KnnClassification(train_data, test_a, k)
        if predicted == test_a[-1]:
            right_predictions += 1

    accuracy = right_predictions / datatest_length * 100
    return right_predictions, datatest_length, accuracy

data = pd.read_csv('diabetes.csv')
data = [list(row) for row in data.values]

# Split the data into training and testing sets
train_data, test_data = SplitData(data, ratio=0.7)

# Min-Max Scaling for both training and test sets
train_data = MinMaxScaling(train_data)
test_data = MinMaxScaling(test_data)

# Specify different values of k for multiple iterations
k_values = [2, 3, 4, 5]

# Initialize variables for average accuracy calculation
total_correct = 0
total_instances = 0

# Iterate over different values of k
for k in k_values:
    correct, total, accuracy = TestKnn(train_data, test_data, k)

    # Output for each iteration
    print(f"\nFor k={k}:")
    print(f"Number of correctly classified instances: {correct}")
    print(f"Total number of instances in the test set: {total}")
    print(f"Accuracy: {accuracy:.2f}%")

    # Update variables for average accuracy calculation
    total_correct += correct
    total_instances += total

# Calculate and output the average accuracy
average_accuracy = total_correct / total_instances * 100
print("\nAverage Accuracy Across All Iterations:", f"{average_accuracy:.2f}%")