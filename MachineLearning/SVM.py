
from sklearn import svm


def LoadFile(filename):
    file = open(filename, "r")
    lines = file.readlines()
    sample = []
    lable = [];

    for line in lines :
        input = str.split(line, ",")
        lst = []
        single_label = []
        num = 0
        index = 0
        for s in input:
            index += 1
            if (index) > 16:
                num = int(s)
                #single_label.append(float(num) / 100.0)
                single_label.append(int(num))
                break
            else:
                lst.append(float(s) / 100.0)
        #lst.append(int(s))
        sample.append(lst)
        lable.append(single_label)
    return sample, lable

if __name__ == '__main__':
    cases, labels = LoadFile("Train.dat")
    test_cases, test_labels = LoadFile("test.txt")
    print("train cases: ", cases)
    print("train labels: ", labels)

    newlables = []
    for arry in labels:
        newlables.append(arry[0])

    clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovo')
    #clf = svm.SVC()
    clf.fit(cases, newlables)

    print(clf.score(cases, newlables))
    result = clf.predict(test_cases)
    print(result)