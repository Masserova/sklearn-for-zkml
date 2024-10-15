from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.datasets import load_iris
from sklearn import tree
import pandas

# fetch aduld dataset
X_adult, y_adult = fetch_openml("adult", version=2, return_X_y=True)

# encode the target and the sex categories via integers
y_adult = y_adult.cat.rename_categories([50,51])
X_adult["sex"]= X_adult["sex"].cat.rename_categories([0,1])

# prepare list of columns that we want to one-hot encode
columns_to_change = list(set(X_adult.columns) - 
                    set(["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week", "sex"]))
# convert categorical data to indicator values
X_adult = pandas.get_dummies(X_adult, dtype=int, columns=columns_to_change)

# multiply to only have integer types
X_adult[X_adult.select_dtypes(include=['number']).columns] *= 10

# only use a part of dataset for now
X_adult = X_adult.head(100)
y_adult = y_adult.head(100)

X_adult.to_csv("adult.csv", index=False, header=True)

#a = array.array('i',(i for i in range(1,105)))


decision_tree = DecisionTreeClassifier(random_state=19, max_depth=6)
decision_tree.fit(X_adult, y_adult)

r = export_text(decision_tree, max_depth=6, decimals=0, spacing=1)

with open('adult_dt.txt', 'w') as output:
    output.write(r)
