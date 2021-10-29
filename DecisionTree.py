import numpy as np

def trainDecisionTree(np_label, np_attrs):

    print("Data shape: " + str(np.shape(np_attrs)))
    # To decide whether an attribute is discrete
    b_discrete = []
    TH_DISCRETE = 10
    for i in range(0,np.shape(np_attrs)[1]):
        s = set()
        b_discrete.append(True)
        col = np_attrs[:,i]
        for x in col:
            s.add(x)
            if(len(s) > TH_DISCRETE):
                b_discrete[-1] = False

    node = TreeNode()
    processTreeNode(node, np_label, np_attrs, b_discrete)
    return node


def compareEqual(left, right):
    return left == right

# def compareNotEqual(left, right):
#     return left != right

def compareLessOrEqual(left, right):
    return left <= right

# def compareBiggerOrEqual(left, right):
#     return left >= right

class TreeNode:
    def __init__(self):
        self.label = None
        self.lChild = None
        self.rChild = None
        self.compareIndexAttr = None
        self.compareValue = None
        self.compareMethod = None

    def accept(self, attrs):
        attr = attrs[self.compareIndexAttr]
        if self.compareMethod(attr, self.compareValue):
            return True
        return False

    def predict(self, attrs):
        if(self.label != None):
            return self.label
        if self.lChild.accept(attrs):
            return self.lChild.predict(attrs)
        else:
            return self.rChild.predict(attrs)

        # Impossible!
        print("TreeNode Error: no child accept!")
        print("arrts is: " + attrs)
        exit(-1)



def devide(np_label, np_attrs, compareIndexAttr, compareMethod, compareValue):
    left_label = []
    left_attrs = []
    right_label = []
    right_attrs = []
    for i in range(0,np.shape(np_attrs)[0]):
        value = np_attrs[i][compareIndexAttr]
        label = np_label[i]
        attr = np_attrs[i]
        if(compareMethod(value, compareValue)):
            left_label.append(label)
            left_attrs.append(attr)
        else:
            right_label.append(label)
            right_attrs.append(attr)
    left_np_label = np.array(left_label)
    left_np_attrs = np.array(left_attrs)
    right_np_label = np.array(right_label)
    right_np_attrs = np.array(right_attrs)
    return left_np_label, left_np_attrs, right_np_label, right_np_attrs

def countDistinctValues(np_values):
    s = dict()
    for v in np_values:
        if v in s:
            s[v] += 1
        else:
            s[v] = 1
    return s


def findDevidePoint(np_label, np_attrs, indexAttr, bDiscrete):
    if bDiscrete:
        compareMethod = compareEqual
        candidateValue = countDistinctValues(np_attrs[:,indexAttr])

    else:
        compareMethod = compareLessOrEqual
        sorted_a = (np_attrs[np_attrs[:,indexAttr].argsort()])[:,indexAttr]

        candidateValue = []
        for i in range(0, len(sorted_a) - 1):
            v = (sorted_a[i] + sorted_a[i+1]) / 2
            candidateValue.append(v)

    minGiniIndex = 1
    for v in candidateValue:
        l_label, l_attr, r_label, r_attr = devide(np_label, np_attrs, indexAttr, compareMethod, v)
        ls_label = [l_label, r_label]
        theGiniIndex = giniIndex(ls_label)
        if theGiniIndex < minGiniIndex:
            minGiniIndex = theGiniIndex
            compareValue = v

    return compareMethod, compareValue, minGiniIndex


def processTreeNode(node, np_label, np_attrs, b_discrete):
    if len(np_label) != len(np_attrs):
        print("Error: label size != attr size")
        exit(-1)
    if len(np_label) <= 0:
        print("Error: label size <= 0!")
        exit(-1)
    if np.shape(np_attrs)[1] != len(b_discrete):
        print("Error: numbers of attrs != length of b_discrete!")
        exit(-1)
    if isArrayElementIdentity(np_label):
        node.label = np_label[0]
        return
    NUM_END = 5;
    if len(np_label) <= NUM_END:
        node.label = getMostElement(np_label)
        return
    if len(np_label) > 1000:
        print("Current recursion data size: " + str(len(np_label)))
    # Find the best attribute to divide.
    minGiniIndex = 1
    # ls_thread = []
    for i in range(0, np.shape(np_attrs)[1]):
        compareMethod, compareValue, giniIndex = findDevidePoint(np_label, np_attrs, i, b_discrete[i])
        if giniIndex < minGiniIndex:
            minGiniIndex = giniIndex
            chooseAttrIndex = i
            chooseCompareMethod = compareMethod
            chooseCompareValue = compareValue


    # Divide the dataset
    l_label, l_attrs, r_label, r_attrs = devide(np_label,
                                                np_attrs,
                                                chooseAttrIndex,
                                                chooseCompareMethod,
                                                chooseCompareValue)
    # Generate subtrees
    node.lChild = TreeNode()
    node.lChild.compareIndexAttr = chooseAttrIndex
    node.lChild.compareMethod = chooseCompareMethod
    node.lChild.compareValue = chooseCompareValue
    if np.shape(l_label)[0] == 0:
        node.lChild.label = getMostElement(np_label)
    else:
        processTreeNode(node.lChild, l_label, l_attrs, b_discrete)
    node.rChild = TreeNode()
    if np.shape(r_label)[0] == 0:
        node.rChild.label = getMostElement(np_label)
    else:
        processTreeNode(node.rChild, r_label, r_attrs, b_discrete)



def isArrayElementIdentity(np_array):
    e = np_array[0]
    for x in np_array:
        if x != e:
            return False
    return True

def getMostElement(np_array):
    dictCount = {}
    for x in np_array:
        if x in dictCount.keys():
            dictCount[x] += 1
        else:
            dictCount[x] = 1
    max = -1
    result = None
    for key in dictCount:
        if dictCount[key] > max:
            result = key
            max = dictCount[key]

    return result

def gini(ls_p):
    result = 1
    for p in ls_p:
        result -= p*p
    return result

def giniIndex(ls_devide_np_label):
    countTotal = 0
    for np_label in ls_devide_np_label:
        countTotal += np.shape(np_label)[0]
    result = 0
    for np_label in ls_devide_np_label:
        countValues = countDistinctValues(np_label)
        ls_p = []
        for v in countValues:
            p = countValues[v] / np.shape(np_label)[0]
            ls_p.append(p)
        result += gini(ls_p) * np.shape(np_label)[0] / countTotal
    return result

