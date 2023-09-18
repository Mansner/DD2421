#!/usr/bin/env python
import monkdata as m
import dtree as d
import matplotlib.pyplot as plt
import numpy as np
import random

# Importing the PyQt5 drawing functionality
from drawtree_qt5 import drawTree  # Replace with the correct import based on where you've saved the PyQt5 code

N = 10  # Initialize N; set it to the number of runs you want for each fraction
fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
datasets = [(m.monk1, m.monk1test), (m.monk3, m.monk3test)]

results = {}

for train_data, test_data in datasets:
    for frac in fractions:
        test_errors = []

        for run in range(N):  # N is the number of runs for statistics
            # Partition the data
            train_subset = random.sample(train_data, int(frac * len(train_data)))
            prune_subset = list(set(train_data) - set(train_subset))

            # Build initial tree
            tree = d.buildTree(train_subset, m.attributes)

            # Find the best pruned tree
            best_tree = tree
            best_score = d.check(tree, prune_subset)

            for pruned_tree in d.allPruned(tree):
                score = d.check(pruned_tree, prune_subset)
                if score > best_score:
                    best_tree = pruned_tree
                    best_score = score

            # Optionally, draw the best pruned tree
            # drawTree(best_tree)  # Comment/uncomment this line to enable/disable tree drawing

            # Evaluate the test error
            test_error = 1 - d.check(best_tree, test_data)
            test_errors.append(test_error)

        # Store the mean and spread of test errors
        results[(train_data, frac)] = (np.mean(test_errors), np.std(test_errors))

# Plotting
monk1_means = []
monk1_stds = []
monk3_means = []
monk3_stds = []

# Extract relevant mean and std for each
for train_data, test_data in datasets:
    means = []
    stds = []
    for frac in fractions:
        mean, std = results[(train_data, frac)]
        means.append(mean)
        stds.append(std)

    # sSplit into values for respective dataset
    if train_data == m.monk1:
        monk1_means = means
        monk1_stds = stds
    else:
        monk3_means = means
        monk3_stds = stds

plt.figure(figsize=(10, 6))
plt.errorbar(fractions, monk1_means, yerr=monk1_stds, fmt='o-', label='MONK1')
plt.errorbar(fractions, monk3_means, yerr=monk3_stds, fmt='x-', label='MONK3')
plt.xlabel('Fraction')
plt.ylabel('Mean Test Error')
plt.title('Effect of Pruning on Test Error for MONK1 and MONK3')
plt.legend()
plt.grid(True)
plt.show()
