from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import matplotlib.pyplot as plt
from scipy.stats import norm
import operator
############################################
from sklearn.model_selection import train_test_split, KFold, cross_val_score  ### used to split the data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, fbeta_score   ### to evaluate the model

from sklearn.model_selection import GridSearchCV

#### Algorithms models to be compared

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
#from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

#from xgboost import XGBClassifier



#Loading the data
df = pd.read_csv("../data/crime_prep2.csv")
print(df.head())
print(df["target"].count())

#looking for missings, kind of data and shape:
print(df.info(verbose=True))
print(df.describe())
print("As count function only takes in account values different to NaN we can use it to drop the columns that have 30% or more of NaN /n")
# df.count() does not include NaN values
df2 = df[[column for column in df if df[column].count() / len(df) >= 0.3]]
print("List of dropped columns:", end=" ")
for c in df.columns:
    if c not in df2.columns:
        print(c, end=", ")
print('\n')
df = df2
print("infromation for new df",df.info(verbose=True))
print("more information about the df", df.describe())
#looking for unique values
print(df.nunique())


#Looking the description of the data
print(df.head())
sns.set(style="darkgrid")
ax1 = sns.countplot(x="target", data=df)
plt.savefig('../report/figures/count_target.png')
#describing the data
print(df['target'].describe())
plt.figure(figsize=(9, 8))
ax2 = sns.distplot(df['target'], color='g', bins=100, hist_kws={'alpha': 0.4})
plt.savefig('../report/figures/description_target.png')


#Getting the numerical data to look into the distribution
list(set(df.dtypes.tolist()))
print("types", list(set(df.dtypes.tolist())))
df_num = df.select_dtypes(include = ['float64', 'int64'])
df_num.head()
print(df_num.head())
# Getting the data that is no numeric
df_obj = df.select_dtypes(exclude=['float64','int64'])
df_obj = [df_num["target"], df_obj]
df_obj = pd.concat(df_obj, axis=1)
print("the other type /n",df_obj.head())

#plotting the histograms for the numerical features
ax3 = df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8); # ; avoid having the matplotlib verbose informations
plt.savefig('../report/figures/features_histo.png')
#from that plotting, we can observe that there are features that have the same
#distribution than our target variable

## Looking into correlation to know which are the features that are strongly
## correlated with the target variable
df_num_corr = df_num.corr()['target'][:-1] # -1 because the latest row is target
golden_features_list = df_num_corr[abs(df_num_corr) > 0.5].sort_values(ascending=False)
print("There is {} strongly correlated values with target:\n{}".format(len(golden_features_list), golden_features_list))
## plotting the features vs the target variable to have visuak information about their relationship
for i in range(0, len(df_num.columns), 5):
    sns.pairplot(data=df_num,
                x_vars=df_num.columns[i:i+5],
                y_vars=['target'])
    plt.savefig('../report/figures/other_relationships_'+ str(i) + '_.png')
 
 ## Removing the 0 values to go deeper in the correlation analysis
individual_features_df = []
for i in range(1, len(df_num.columns)): # because the first element is the target variable
    tmpDf = df_num[[df_num.columns[i], 'target']]
    tmpDf = tmpDf[tmpDf[df_num.columns[i]] != 0]
    individual_features_df.append(tmpDf)
all_correlations = {feature.columns[0]: feature.corr()['target'][0] for feature in individual_features_df}
all_correlations = sorted(all_correlations.items(), key=operator.itemgetter(1))
for (key, value) in all_correlations:
    print("{:>15}: {:>15}".format(key, value))

## From that correlation analysis, we can get a list of the strongest correlated features
    
golden_features_list = [key for key, value in all_correlations if abs(value) >= 0.5]
print("There are {} strongly correlated values with target:\n{}".format(len(golden_features_list), golden_features_list))


## Feature - feature correlation

corr = df_num.drop('target', axis=1).corr() # We already examined target  correlations
plt.figure(figsize=(12, 10))

sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)],
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True)
plt.savefig('../report/figures/heatmap_correlations.png')
df_quantitative_values = df_num
df_quantitative_values.head()
print(df_quantitative_values.head())

features_to_analyse = [x for x in list(df_quantitative_values.columns) if x in golden_features_list]
features_to_analyse.append('target')
features_to_analyse
print("Features to analyse: ", features_to_analyse)

fig, ax = plt.subplots(round(len(features_to_analyse) / 3), 3, figsize = (18, 12))

for i, ax in enumerate(fig.axes):
    if i < len(features_to_analyse) - 1:
        sns.regplot(x=features_to_analyse[i],y='target', data=df[features_to_analyse], ax=ax)
    plt.savefig('../report/figures/distribution_vs_target'+ str(i) + '_.png')
    
plt.figure(figsize = (10, 6))
ax = sns.boxplot(x='v_cat_2', y='target', data=df_obj)
plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")
plt.xticks(rotation=45)
plt.savefig('../report/figures/non_numerical_distributions.png')

################################################################

### Creating X and y variables

features_to_analyse_df = df_num.filter(features_to_analyse, axis=1)
features_to_analyse_df = features_to_analyse_df.fillna(0)
print("features to analyse /n", features_to_analyse_df.head())

features_to_analyse_df = pd.get_dummies(features_to_analyse_df)  # convert categorical to one-hone encoding

features_to_analyse_df.sample(frac=1)  # shuffle data

X = features_to_analyse_df.drop(['target'], axis=1)
y = features_to_analyse_df['target']

# convert to numpy arrays
X = X.to_numpy()
Y = y.to_numpy()

# split into training and testing sets
n = len(X)

train_perc = 0.75  # percentage of training set
train_ind = range(0, int(train_perc*n))  # indices of dataset for training
train_x = X[train_ind]
#train_y = np.log(Y[train_ind])  # perform logarithm transformation
train_y = Y[train_ind]

test_ind = range(n-int(train_perc*n), n)  # indices of dataset for testing
test_x = X[test_ind]
#test_y = np.log(Y[test_ind])  # perform logarithm transformation
test_y = Y[test_ind]
######################################################

# create model
model = LinearRegression()
# calculate beta using train
model.fit(train_x, train_y)

err = []  # calculate MSE error
err.append(mean_squared_error(test_y, model.predict(test_x)))  # test MSE
err.append(mean_squared_error(train_y, model.predict(train_x)))  # train MSE
plt.barh(['Test', 'Train'], err)
for i, v in enumerate(err):
    plt.text(v, i, str(np.round(v, 3)))
plt.ylabel("MSE")
plt.suptitle("Linear Regression MSE for Test and Train Data")
plt.savefig('../report/figures/LR_MSE_for_Test_and_train_data.png')
#plt.show()

err = []  # calculate R^2 error
err.append(r2_score(test_y, model.predict(test_x)))  # test MSE
err.append(r2_score(train_y, model.predict(train_x)))  # train MSE
plt.barh(['Test', 'Train'], err)
for i, v in enumerate(err):
    plt.text(v, i, str(np.round(v, 3)))
plt.ylabel("R^2")
plt.suptitle("Linear Regression R^2 for Test and Train Data")
plt.savefig('../report/figures/LR_R2_for_test_and_train_data.png')
#plt.show()

residuals = model.predict(train_x)-train_y
plt.scatter(train_y, residuals)
plt.xlabel("Target Variable: Y")
plt.ylabel("Residuals: F(x)-Y")
plt.suptitle("Residual Plot for Linear Regression")
plt.axhline(0, color='black')
#plt.show()
plt.savefig('../report/figures/residual_plot_for_LR.png')

residuals = np.power(model.predict(train_x)-train_y, 2)
plt.scatter(train_y, residuals)
plt.xlabel("Target Variable: Y")
plt.ylabel("Squared Residuals: (F(x)-Y)^2")
plt.suptitle("Squared Residual Plot for Linear Regression")
plt.axhline(0, color='black')
#plt.show()
plt.savefig('../report/figures/square_residual_plot_forLR.png')
        





