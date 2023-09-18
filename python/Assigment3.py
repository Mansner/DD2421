import monkdata as m
import dtree as d

# Calculate information gain for each attribute
datasets = [m.monk1, m.monk2, m.monk3]
for dataset in datasets:
    gains = []
    for i in range(6):
        gain = d.averageGain(dataset, m.attributes[i])
        gains.append(gain)
        print(f"Information Gain for attribute {i + 1}: {gain}")
    best_attribute_index = gains.index(max(gains))
    best_attribute = m.attributes[best_attribute_index]
    print(f"The best attribute for splitting is: {best_attribute}")

# Determine the attribute with the highest information gain

