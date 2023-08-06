from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn import datasets
# print("96.07")
# load the dataset
data = datasets.load_iris()
X = data.data
y = data.target

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create an Intel oneAPI pipeline with PCA and logistic regression
pipeline = make_pipeline(PCA(n_components=2), LogisticRegression())

# train the model
pipeline.fit(X_train, y_train)

# make predictions on the test set
y_pred = pipeline.predict(X_test)

# calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)