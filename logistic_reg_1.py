import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.pandas.set_option('display.max_column',None) # To display all column values in dataset
#read data from data.csv file
df = pd.read_csv('data.csv')
print(df.head())
print(df.info())
print(df.size)
print(df.shape)
print(df['diagnosis'].unique())

df = df.drop('id',axis=1)
df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})
plt.figure(figsize=(20,20))
sns.heatmap(df.corr(),annot=True)
plt.show()
# to check data is balanced or not
print(df['diagnosis'].value_counts())

#selecting the independent columns using SelectKBest and chi2 methods from sklearn library

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X = df.drop('diagnosis',axis=1)  #independent columns
y = df['diagnosis']    #target column i.e price range
#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))  #print 10 best features

#feature selection of independent columns using ExtraTreesClassifier

from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()
y = df['diagnosis']


print(feat_importances.nlargest(10).index)

#model building

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error,confusion_matrix
X = df[['perimeter_worst', 'area_worst', 'concave points_mean', 'area_mean',
       'perimeter_mean', 'radius_worst', 'concave points_worst', 'radius_mean',
       'concavity_mean', 'concavity_worst']]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
df_new = pd.DataFrame({'old':y_test,'new':y_pred})
print(df_new.head())


print('MSE',mean_squared_error(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
