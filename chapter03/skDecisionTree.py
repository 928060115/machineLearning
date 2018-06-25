# -*- coding:utf-8 -*-
"""
  @author:ly
  @file: skDecisionTree.py
  @time: 2018/6/2517:11
  @version: v1.0
  @Dec: 使用sklearn实现决策树
"""

from sklearn import tree
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import pydotplus
from sklearn.externals.six import StringIO

if __name__ == '__main__':
    with open('data/lenses.txt') as fr:
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
        print(lenses)
        lenses_target = []
        for each in lenses:
            lenses_target.append(each[-1])

        lensesLabels = ['age','prescript','astigmatic','terRate']
        lenses_list = []
        lenses_dict = {}
        for each_label in lensesLabels:
            for each in lenses:
                lenses_list.append(each[lensesLabels.index(each_label)])
            lenses_dict[each_label] = lenses_list
            lenses_list = []
        print(lenses_dict)
    lenses_pd = pd.DataFrame(lenses_dict)
    print(lenses_pd)
    le = LabelEncoder()
    for col in lenses_pd.columns:
        lenses_pd[col] = le.fit_transform(lenses_pd[col])
    print(lenses_pd)
    clf = tree.DecisionTreeClassifier(max_depth=4,criterion='entropy')
    clf = clf.fit(lenses_pd.values.tolist(),lenses_target)
    dot_data = StringIO()
    tree.export_graphviz(clf,out_file=dot_data,feature_names=lenses_pd.keys(),
                         class_names=clf.classes_,filled=True,rounded=True,special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf('data/tree1.pdf')
    print(clf.predict([[1,1,1,0]]))