import math
import pandas as pd
import numpy as np


# for getting stats like mean , standard deviation and probability of each class
def stats(data):

    # scaling over zero and one by normalization method
    data = (data - data.min()) / (data.max() - data.min())

    mean_by_class = data.groupby([status_col]).mean()
    std_by_class = data.groupby([status_col]).std()

    # value_counts return the values by category
    # here we calculate the probability of each status
    prob_by_class = data[status_col].value_counts() / len(data.index)


    return mean_by_class, std_by_class, prob_by_class, data

    # this function is useful for calculating likehood
    # likelihood is P(data|class)


def gaussian(x, mean, std):
    exponent = math.exp(-((x - mean) ** 2 / (2 * std ** 2)))
    return (1 / (math.sqrt(2 * math.pi) * std)) * exponent

    # here we finally calculating probability of each class and
    # comparing which one is greater then returning that class
    # its my task1


def calculate_probability(row, mean_by_class, std_by_class, prob_by_class):
    probs = dict()

    # first we get classes and iterate through it here we have only 2 classes [0,1]
    for classs in mean_by_class.index.values:
        # implementing formula for gaussian naive bayes theorem
        probability = prob_by_class[classs]
        for col_numb in np.arange(len(mean_by_class.loc[classs])):
            probability *= gaussian(row[col_numb], mean_by_class.loc[classs][col_numb],
                                    std_by_class.loc[classs][col_numb])
        probs[classs] = probability
    if probs.get(0) > probs.get(1):
        return 0
    else:
        return 1


# this method compare actual values with predicted for accuracy check of our model
# my task 2
def compare(predicted, actual):
    count = 0
    for i in np.arange(len(predicted)):
        if predicted[i] == actual[i]:
            count += 1
    return count / len(predicted) * 100


# load data set
df_train = pd.read_excel("parktraining.xlsx", header=None)
df_test = pd.read_excel("parktesting.xlsx", header=None)

dfs = {"train": df_train, "test": df_test}

# column number 22 is status column
# i did not wanted to give headers to data frame so i am just initializing it by looking at data
status_col = 22

# current dataset (its just for making sure that you putted same data source in all fields like (accuracy, model etc) )
#current_df = df_train

# Model training

for current_df in dfs:

    final_accuracy = 0
    count_for_iterations = 0

    # dividing our dataset into sub datasets so we can training again and again.
    for g, df in dfs[current_df].groupby(np.arange(len(dfs[current_df])) // 70):
        mean_by_class, std_by_class, prob_by_class, df  = stats(df)
        predictions = list()
        # here we are classifying each row
        for index, row in df.iterrows():
            prediction = calculate_probability(row, mean_by_class, std_by_class, prob_by_class)
            predictions.append(prediction)

        # check accuracu during each iteration
        accuracy = compare(predictions, df[status_col].tolist())
        print("Accuracy during round {}: {}".format(g+1, accuracy))
        final_accuracy += accuracy
        count_for_iterations = g

    print("Ground truth values for " + current_df + " data set are : ")
    print(df[status_col].tolist())
    print("Output labels (which is predicted from model) for " + current_df + " data set are : ")
    print(predictions)
    print("when we compare ground truth values with predicted value then we get total accuracy of Model.")
    print("Model accuracy for " + current_df + " data set set is " + str(final_accuracy/(count_for_iterations+1)) + "%.")
    print("-----------------------------------------------------------")



