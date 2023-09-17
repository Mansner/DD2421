import monkdata as m
import dtree as d

# Create full decision trees for each MONK dataset
tree_monk1 = d.buildTree(m.monk1, m.attributes)
tree_monk2 = d.buildTree(m.monk2, m.attributes)
tree_monk3 = d.buildTree(m.monk3, m.attributes)

# Check the performance of each tree on its respective training set
train_performance_monk1 = d.check(tree_monk1, m.monk1)
train_performance_monk2 = d.check(tree_monk2, m.monk2)
train_performance_monk3 = d.check(tree_monk3, m.monk3)

# Calculate the error on the training set
train_error_monk1 = 1 - train_performance_monk1
train_error_monk2 = 1 - train_performance_monk2
train_error_monk3 = 1 - train_performance_monk3

# Check the performance of each tree on its respective test set
test_performance_monk1 = d.check(tree_monk1, m.monk1test)
test_performance_monk2 = d.check(tree_monk2, m.monk2test)
test_performance_monk3 = d.check(tree_monk3, m.monk3test)

# Calculate the error on the test set
test_error_monk1 = 1 - test_performance_monk1
test_error_monk2 = 1 - test_performance_monk2
test_error_monk3 = 1 - test_performance_monk3

# Print out results
print(f"Train performance for MONK-1: {train_performance_monk1}, Train error: {train_error_monk1}")
print(f"Test performance for MONK-1: {test_performance_monk1}, Test error: {test_error_monk1}")
print(f"Train performance for MONK-2: {train_performance_monk2}, Train error: {train_error_monk2}")
print(f"Test performance for MONK-2: {test_performance_monk2}, Test error: {test_error_monk2}")
print(f"Train performance for MONK-3: {train_performance_monk3}, Train error: {train_error_monk3}")
print(f"Test performance for MONK-3: {test_performance_monk3}, Test error: {test_error_monk3}")
