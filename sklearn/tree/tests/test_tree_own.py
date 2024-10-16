from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.datasets import load_iris
from sklearn.preprocessing import OrdinalEncoder
from sklearn import tree
import pandas
import array
import numpy as np

# fetch aduld dataset
X_adult, y_adult = fetch_openml("adult", version=2, return_X_y=True)
X_adult.to_csv("adult_initial.csv", index=False, header=False)
X_adult["sex"].to_csv("adult_scolumn.csv", index=False, header=False)
# encode the target and the sex categories via integers
y_adult = y_adult.cat.rename_categories([50,51])
X_adult["sex"]= X_adult["sex"].cat.rename_categories([0,1])

# prepare list of columns that we want to one-hot encode
#columns_to_change = list(set(X_adult.columns) - 
                #    set(["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week", "sex"]))
# convert categorical data to indicator values
#X_adult = pandas.get_dummies(X_adult, dtype=int, columns=columns_to_change)


encoder = OrdinalEncoder(dtype=np.int64, encoded_missing_value=-1).set_output(transform="pandas")
X_adult = encoder.fit_transform(X_adult)
# multiply to only have integer types
#X_adult[X_adult.select_dtypes(include=['number']).columns] *= 10


X_adult.to_csv("X_adult.csv", index=False, header=True)
y_adult.to_csv("y_adult.csv", index=False, header=True)

a = array.array('i',(i for i in range(1,15)))

decision_tree = DecisionTreeClassifier(random_state=19, max_depth=10, with_fairness=True, f_threshold=0.2, s_attribute=3)
decision_tree.fit(X_adult, y_adult)

r = export_text(decision_tree, max_depth=10, feature_names=a, decimals=0, spacing=1)

with open('adult_dt.txt', 'w') as output:
    output.write(r)

X_adult = X_adult.assign(goal=y_adult)
print(X_adult.columns.get_loc("sex"))
X_adult.to_csv("adult_full.csv", index=False, header=False)