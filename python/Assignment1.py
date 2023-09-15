import dtree
import monkdata as m

datasets = [m.monk1, m.monk2, m.monk3]

for dataset in datasets:
    print(dtree.entropy(dataset))
