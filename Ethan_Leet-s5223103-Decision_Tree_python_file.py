""" 
Ethan Leet
s5223103
2802 ICT
Assignment 2
Task 2 - Decision-Tree
"""

import pandas as pd
import numpy as np
from numpy import log2 as log
from math import sqrt
import matplotlib.pyplot as plt

eps = np.finfo(float).eps

data = pd.read_csv("votes.csv")

all_data = data.iloc[:]

shuffle_index = np.random.permutation(all_data.shape[0])
all_data = all_data.iloc[shuffle_index]

# split data function takes a percentage and sets up dataframes, values and true class values
def split_data(percentage):
    training_size = int(all_data.shape[0] * percentage)
    training_df = all_data.iloc[:training_size, :].reset_index(drop=True)
    testing_df = all_data.iloc[training_size:, :].reset_index(drop=True)
    training = training_df.values
    testing = testing_df.values
    y_true = testing[:, 0]
    return training_df, testing_df, training, testing, y_true

# initialise dataframes, value and true class values based on 70% training set size
training_df, testing_df, training, testing, y_true = split_data(0.7)

# this function finds the entropy for the entire dataset
def find_entropy(df):
    Class = df.keys()[0]
    entropy = 0
    values = df[Class].unique()
    for value in values:
        fraction = df[Class].value_counts()[value]/len(df[Class])
        entropy += -fraction*np.log2(fraction)
    return entropy

# this function finds entropy for each attribute
def find_entropy_attribute(df, attribute):
    Class = df.keys()[0]
    target_variables = df[Class].unique()
    variables = df[attribute].unique()
    entropy2 = 0
    for variable in variables:
        entropy = 0
        for target_variable in target_variables:
            num = len(df[attribute][df[attribute] == variable]
                      [df[Class] == target_variable])
            den = len(df[attribute][df[attribute] == variable])
            fraction = num/(den+eps)
            entropy += -fraction*log(fraction+eps)
        fraction2 = den/len(df)
        entropy2 += -fraction2*entropy
    return abs(entropy2)

# this function uses the dataset entropy and attribute entropy to work out information gain
def information_gain(df):
    info_gain = []
    for key in df.keys()[1:]:
        info_gain.append(find_entropy(df)-find_entropy_attribute(df, key))
    return df.keys()[1:][np.argmax(info_gain)]

# this function recursively builds a tree
def build_tree(data, originaldata, features, target_attribute_name="party", parent_node_class=None):
    # base cases
    # stop if pure subset
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    # stop if at end of data
    elif len(data) == 0:
        return np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])]
    # stop if no more features left to expand
    elif len(features) == 0:
        return parent_node_class
    # recursively build tree
    else:
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(
            np.unique(data[target_attribute_name], return_counts=True)[1])]
        item_values = [information_gain(data) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        tree = {best_feature: {}}
        features = [i for i in features if i != best_feature]
        for value in np.unique(data[best_feature]):
            value = value
            sub_data = data.where(data[best_feature] == value).dropna()
            subtree = build_tree(sub_data, data, features,
                                 target_attribute_name, parent_node_class)
            tree[best_feature][value] = subtree
        return(tree)

# there are cases that the training set hasnt seen the attribute passed by testing set
# we give it a default value of 1 to seperate this from classified data
def classify(query, tree, default=1):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]]
            except:
                return default
            result = tree[key][query[key]]
            if isinstance(result, dict):
                return classify(query, result)
            else:
                return result

# this function finds the accuracy of the DT classifier
def dt_accuracy(data, tree):
    queries = data.iloc[:, :-1].to_dict(orient="records")
    predicted = []
    for i in range(len(data)):
        predicted.append(classify(queries[i], tree))
    accuracy = (np.sum(predicted == data["party"])/len(data))*100
    return accuracy, predicted

# build the training tree
training_tree = build_tree(training_df, training_df, training_df.columns[1:])
# assign variables from accuracy function
DT_accuracy, prediction = dt_accuracy(testing_df, training_tree)
# initialise confusion matrix
cm = {'y_Actual': testing_df['party'], 'y_Predicted': prediction}
df = pd.DataFrame(cm)
# build confusion matrix
confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=[
                               'Actual'], colnames=['Predicted'], margins=True)
print(confusion_matrix)
# assign variables based on matrix
DT_TN = confusion_matrix['democrat']['democrat']
DT_TP = confusion_matrix['republican']['republican']
DT_FP = confusion_matrix['republican']['democrat']
DT_FN = confusion_matrix['democrat']['republican']
# work out recall/precision/f1 values
DT_recall = DT_TP/(DT_FN+DT_TP)
DT_precision = DT_TP/(DT_FP+DT_TP)
DT_f1 = 2 * ((DT_precision * DT_recall) / (DT_precision + DT_recall))
# print results
print("Decision-Tree accuracy:", DT_accuracy, "%")
print("Decision-Tree precision:", DT_precision)
print("Decision-Tree recall:", DT_recall)
print("Decision-Tree f1:", DT_f1)


# euclidean distance function to calculate kNN
def euclidean_distance(x_testing, x_training):
    distance = 0
    for i in range(1, len(x_testing)):
        distance += (x_testing[i]-x_training[i])**2
    return sqrt(distance)

# calculate kNN
def kNN(x_testing, x_training, num_neighbors):
    distances = []
    data = []
    for i in x_training:
        distances.append(euclidean_distance(x_testing, i))
        data.append(i)
    distances = np.array(distances)
    data = np.array(data)
    sort_indexes = distances.argsort()
    data = data[sort_indexes]
    return data[:num_neighbors]

# classify new data
def classification(x_testing, x_training, num_neighbors):
    classes = []
    neighbors = kNN(x_testing, x_training, num_neighbors)
    for i in neighbors:
        classes.append(i[0])
    predicted = max(classes, key=classes.count)
    return predicted

# work out accuracy based on true vs predicted
def accuracy(true, pred):
    TP = TN = FP = FN = num_correct = 0
    for i in range(len(true)):
        if true[i] == pred[i] == 'democrat':
            num_correct += 1
            TN += 1
        elif true[i] == pred[i] == 'republican':
            num_correct += 1
            TP += 1
        elif true[i] == 'republican' and pred[i] == 'democrat':
            FN += 1
        else:
            FP += 1
    accuracy = (num_correct/len(true)) * 100
    return accuracy, TP, TN, FP, FN


# initialise predicted values
k = 10
y_pred = []
for i in testing:
    y_pred.append(classification(i, training, k))
# set up variables
KNN_accuracy, KNN_TP, KNN_TN, KNN_FP, KNN_FN = accuracy(y_true, y_pred)
# calculate recall/precision/f1 values
KNN_recall = KNN_TP/(KNN_FN+KNN_TP)
KNN_precision = KNN_TP/(KNN_FP+KNN_TP)
KNN_f1 = 2 * ((KNN_precision * KNN_recall) / (KNN_precision + KNN_recall))
# print results
print("\nk-NN accuracy: ", KNN_accuracy, "%")
print("k-NN precision:", KNN_precision)
print("k-NN recall:", KNN_recall)
print("k-NN f1:", KNN_f1)

# following block works out how accuracy changes in terms of training data for kNN
# the result is used to create the learning curve
knn_set_size = []
knn_accurancy_variance = []
for i in range(len(testing_df), len(data), 7):
    knn_set_size.append(i)
    knn_training_set = data.loc[:i]
    knn_training_set_values = knn_training_set.values
    knn_test_predictions = []
    knn_testing_actual = []
    for j in testing:
        knn_test_predictions.append(
            classification(j, knn_training_set_values, 10))
        knn_testing_actual.append(j[0])
    knn_variance, KNN_TP, KNN_TN, KNN_FP, KNN_FN = accuracy(
        knn_testing_actual, knn_test_predictions)
    knn_accurancy_variance.append(knn_variance)

# following block works out how accuracy changes in terms of training data for decision tree
# the result is used to create the learning curve
dt_set_size = []
dt_accurancy_variance = []
for i in range(len(testing_df), len(data), 7):
    dt_set_size.append(i)
    dt_training_set = data.loc[:i]
    dt_test_predictions = []
    dt_testing_actual = testing_df['party']
    growing_tree = build_tree(
        dt_training_set, dt_training_set, dt_training_set.columns[1:])
    acc, pred = dt_accuracy(testing_df, growing_tree)
    acc_list = []
    acc_list.append(acc)
    dt_variance, KNN_TP, KNN_TN, KNN_FP, KNN_FN = accuracy(
        dt_testing_actual, pred)
    dt_accurancy_variance.append(dt_variance)

# plot learning curve
plt.plot(dt_set_size, dt_accurancy_variance, label='Decision-Tree')
plt.plot(knn_set_size, knn_accurancy_variance, label='k-NN')
plt.legend()
plt.xlabel("Training set size values")
plt.ylabel("Accuracy (%)")
plt.title("Learning Curve")
plt.show()
