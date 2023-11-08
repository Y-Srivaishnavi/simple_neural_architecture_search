import pandas as pd
from randomNAS import randomsearch
from sklearn.model_selection import train_test_split

# read the df
df= pd.read_csv('Neural-Architecture-Search-with-RANDOM-SEARCH-main\\splice\\dna.csv')

# split df for training and testing
train_df, test_df = train_test_split(df, test_size=0.3, shuffle=True, random_state=34)

# Preprocess data
train_df['class'] = train_df['class'] - 1
test_df['class'] = test_df['class'] -1

# split it into X and y values
X_train=train_df.drop('class', axis=1)
y_train=train_df['class']

X_test=test_df.drop('class', axis=1)
y_test=test_df['class']

# let the search begin
randomsearch(X_train, y_train, X_test, y_test)
