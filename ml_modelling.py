#final modelling script
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import seaborn as sns
from pprint import pprint
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import seaborn as sns
import pickle

df = pd.read_csv("dataset_500000.csv")
vdf = pd.read_csv("Vaihingen3D_Traininig.csv")


n = 500001

#subsetting only until certain rows because the feature set would be NaNs
df = df.iloc[:n,:]
df["Intensity"] = vdf.loc[:n,"Intensity"]
df["return_number"] = vdf.loc[:n,"return_number"]
df["number_of_returns"] = vdf.loc[:n,"number_of_returns"]
df["label"] = vdf.loc[:n,"label"]




def clean(x):
    
    if isinstance(complex(x), complex):
        return(complex(x).real)
    else:
        return(float(x))


columns_to_clean = ["lambda1","lambda2","lambda3","sum_of_evs","eigenotropy","omnivariance","anisotropy","planarity","linearity","surface_variation","sphericity"]
for column in columns_to_clean:
    df[column] = df[column].fillna(0)
    df[column] = df[column].replace(['inf','-inf','#NAME?'],0)
    df[column] = df[column].apply(clean)



df.to_csv("cleaned_dataset_"+str(n)+".csv", index=False)



#Exploratory data analysis

#barplot of Label vs Mean Intensity
fig, ax = subplots()
fig_df = pd.DataFrame()
fig_df["labels"] = ["Powerline","Low vegetation","Impervious surfaces","Car","Fence/Hedge","Roof","Facade","Shrub","Tree"]

fig_df["count"] = (df.groupby('label', as_index=False)['Intensity'].count())["Intensity"]
fig_df["sum_of_intensities"] = (df.groupby('label', as_index=False)['Intensity'].sum())["Intensity"]
fig_df["mean"] = (df.groupby('label', as_index=False)['Intensity'].mean())["Intensity"]

ax = fig_df.plot.bar(rot=90, x='labels', y='mean')
ax.legend(["Mean of intensities"])
plt.show()


#barplot of Label vs Mean height
fig1_df =  pd.DataFrame()
fig_df = pd.DataFrame()
fig_df["labels"] = ["Powerline","Low vegetation","Impervious surfaces","Car","Fence/Hedge","Roof","Facade","Shrub","Tree"]

fig_df["count"] = (df.groupby('label', as_index=False)['Z'].count())["Z"]
fig_df["sum_of_heights"] = (df.groupby('label', as_index=False)['Z'].sum())["Z"]
fig_df["mean"] = (df.groupby('label', as_index=False)['Z'].mean())["Z"]

ax = fig_df.plot.bar(rot=90, x='labels', y='mean')
ax.legend(["Mean of Heights"])
plt.show()




color_dict = {0:"#006400",1:"#00008b",2:"#b03060",3:"#ff0000",4:"#ffd700",5:"#7cfc00",6:"#00ffff",7:"#ff00ff",8:"#FC7753",9:"#ffdab9"}
labels = ["Powerline","LowVegetation","Impervious surfaces","Car","Fence or Hedge","Roof","Facade","Shrub","Tree"]
fig, ax = plt.subplots(figsize=(10,10))
ax = sns.scatterplot(x='global_height_above',y='Intensity',hue = 'label',data = df,
                     palette = color_dict)
ax.legend(labels)
plt.show()



#Boxplot of label and heights
sns.boxplot(x="label", y="Z", data=df)
plt.show()



#Boxplot of label and heights
sns.boxplot(x="label", y="sum_of_evs", data=df)
plt.show()



#barplot of class distribution
fig, ax = subplots()
fig_df = pd.DataFrame()
fig_df["count"] = df['label'].value_counts().values
labels_dict =  {0:"Powerline",1:"Low vegetation",2:"Impervious surfaces",3:"Car",4:"Fence/Hedge",5:"Roof",6:"Facade",7:"Shrub",8:"Tree"}
fig_df["labels"] = [labels_dict[i] for i in df['label'].value_counts().index]
ax = fig_df.plot.bar(rot=90, x='labels', y='count')
ax.legend(["Label distribution - dataset"])
plt.show()




#extract separate, large train and test and perform over it
X = df.iloc[:,:-1]
y = df.iloc[:,-1:]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42) #


#Please uncomment only if you want to perform hyper-parameter tuning. This will take close to 1.5hrs for complete execution.
""" 
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 200)]
# Number of features to consider at every split
max_features = ['auto']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid


random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'bootstrap': bootstrap}


pprint(random_grid)


rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(X_train, y_train)

pprint(rf_random)
print("*"*20)
print(rf_random.cv_results_)
print("*"*20)
print(rf_random.best_estimator_)
print("*"*20)
print(rf_random.best_params_)
"""


import matplotlib.pyplot as plt

kf = KFold(n_splits=5)
for train, validation in kf.split(X_train):
    X_train_cv = X_train.iloc[train,:-1]
    y_train_cv = y_train.iloc[train,-1:]
    X_val_cv = X_train.iloc[validation,:-1]
    y_val_cv = y_train.iloc[validation,-1:]

    clf = RandomForestClassifier(n_estimators=1000, max_features="auto", max_depth=90, bootstrap=True)
    clf.fit(X_train_cv, y_train_cv)
    y_pred_cv = clf.predict(X_val_cv)

    print(classification_report(y_val_cv, y_pred_cv))
    print(X_train_cv.columns)
    print(clf.feature_importances_)
    plt.show(pd.Series(clf.feature_importances_, index=X_train_cv.columns)
    .plot(kind='barh'))


#final training
print("--------------------- FINAL TRAINING AND TESTING ---------------------------- ")
rf_model = RandomForestClassifier(n_estimators=1000, max_features="auto", max_depth=90, bootstrap=True)
rf_model.fit(X_train, y_train)

#saving the trained random forest model to a pickle file
filename = 'random_forest_model.sav'
pickle.dump(rf_model, open(filename, 'wb'))

y_pred_final = rf_model.predict(X_test)
print("Final classification report")
print(classification_report(y_test, y_pred_final))


feat_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
feat_importances.nlargest(25).plot(kind='barh')





