import numpy as np
import DecisionTree
from sklearn import datasets

def main():
    # Test the accuracy of our decision tree
    print("\nloading dataset: slearn's Breast Cancer")
    dataset = datasets.load_breast_cancer()
    train_set, test_set = splitData(dataset)
    node = DecisionTree.trainDecisionTree(train_set[:, -1], train_set[:, :-1])
    print("Finish training. Now testing...")
    print("Accuracy on sklearn's Breast Cancer dataset: "
          + str(test(node, test_set[:,-1], test_set[:,:-1])))

    # Test in another dataset
    trainAndTest(datasets.load_iris, "sklearn's Iris")
    trainAndTest(datasets.load_wine, "sklearn's Wine")



def splitData(dataset):
    data = dataset.data
    target = dataset.target
    m_data = np.mat(data)
    m_target = np.mat(target).transpose()
    m_dataset = np.c_[m_data, m_target]
    dataset = np.array(m_dataset)
    train_set = []
    test_set = []

    for i in range(0, len(data)):
        if i % 2 == 0:
            test_set.append(dataset[i])
        else:
            train_set.append(dataset[i])
    train_set = np.array(train_set)
    test_set = np.array(test_set)

    return train_set, test_set

def test(node, np_label, np_attrs):
    accuracy = 0
    for i in range(0,np.shape(np_attrs)[0]):
        yp = node.predict(np_attrs[i])
        # print("predict:",yp)
        # print("real value:",np_label[i])
        if yp == np_label[i]:
            accuracy += 1 / np.shape(np_attrs)[0]
    return accuracy

def trainAndTest(funGetDataset, name):
    print()
    print("loading dataset: " + name)
    dataset = funGetDataset()
    print("Training...")
    train_set, test_set = splitData(dataset)
    node = DecisionTree.trainDecisionTree(train_set[:, -1], train_set[:, :-1])
    test_np_label = test_set[:, -1]
    test_np_attrs = test_set[:, :-1]
    print("Finish training. Now testing...")
    print("Accuracy on " + name  + " dataset: "
          + str(test(node, test_np_label, test_np_attrs)))

if __name__ == "__main__":
    main()
