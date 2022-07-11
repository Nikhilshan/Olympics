import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def preprocess(df,region_df):
    # filtering for summer olympics
    df = df[df['Season'] == 'Summer']
    # merge with region_df
    df = df.merge(region_df, on='NOC', how='left')
    # dropping duplicates
    df.drop_duplicates(inplace=True)
    # one hot encoding medals
    df = pd.concat([df, pd.get_dummies(df['Medal'])], axis=1)
    return df
def his_preprocessor(olympic_df):
    data_df = olympic_df
    data_df.drop('Year', axis=1, inplace=True)
    olympic_df = pd.read_csv('Historical_Olympics_data.csv')
    data_df = data_df.groupby(['Country'])[
        ['Host', 'Athletes', 'Sports', 'Events', 'Gold', 'Silver', 'Bronze', 'Medals']].sum().reset_index()
    data_df = data_df.sort_values(by='Medals', ascending=False)
    data_df['CountryId'] = [(x + 1) for x in range(211)]
    data_df = data_df[['Country', 'CountryId']]
    olympic_df = pd.merge(olympic_df, data_df, how='left', on='Country')
    olympic_df['Athletes per sport'] = round(olympic_df['Athletes'] / olympic_df['Sports'], 2).replace(np.inf, 0)
    # Bring 'CountryId' column to the front
    col = olympic_df.pop('CountryId')
    olympic_df.insert(0, 'CountryId', col)
    olympic_df = olympic_df.sort_values(['Year', 'Medals', 'Country'], ascending=[True, False, True])
    olympic_df = pd.get_dummies(olympic_df, columns=['Country'])
    predict_year = 2022
    train_df = olympic_df[olympic_df['Year'] < predict_year]
    X = train_df.drop(['Gold', 'Silver', 'Bronze', 'Medals'], axis=1)

    y1 = train_df['Gold'].values.reshape(-1, 1)
    y2 = train_df['Silver'].values.reshape(-1, 1)
    y3 = train_df['Bronze'].values.reshape(-1, 1)
    y4 = train_df['Medals'].values.reshape(-1, 1)
    test_year=2020
    test_df = olympic_df[olympic_df['Year'] == test_year].sort_values(['CountryId'])
    test_data = test_df.drop(['Gold', 'Silver', 'Bronze', 'Medals'], axis=1).reset_index(drop=True)

    # Gold
    X1_train, X1_test, y1_train, y1_test = train_test_split(X, y1, random_state=50)

    # Silver
    X2_train, X2_test, y2_train, y2_test = train_test_split(X, y2, random_state=50)

    # Bronze
    X3_train, X3_test, y3_train, y3_test = train_test_split(X, y3, random_state=50)

    # Total Medals
    X4_train, X4_test, y4_train, y4_test = train_test_split(X, y4, random_state=50)

    model1 = LinearRegression()
    model1.fit(X1_train, y1_train)

    model2 = LinearRegression()
    model2.fit(X2_train, y2_train)

    model3 = LinearRegression()
    model3.fit(X3_train, y3_train)

    model4 = LinearRegression()
    model4.fit(X4_train, y4_train)

    gold_predictions = model1.predict(test_data)
    gold_predictions = np.ravel(gold_predictions)
    gold_predictions = np.around(gold_predictions, decimals=0).astype(int)

    silver_predictions = model2.predict(test_data)
    silver_predictions = np.ravel(silver_predictions)
    silver_predictions = np.around(silver_predictions, decimals=0).astype(int)

    bronze_predictions = model3.predict(test_data)
    bronze_predictions = np.ravel(bronze_predictions)
    bronze_predictions = np.around(bronze_predictions, decimals=0).astype(int)

    for i in range(0, len(gold_predictions)):
        if gold_predictions[i] < 0:
            gold_predictions[i] = 1
        if silver_predictions[i] < 0:
            silver_predictions[i] = 1

    total_medals_predictions = model4.predict(test_data)
    total_medals_predictions = np.ravel(total_medals_predictions)
    total_medals_predictions = np.around(total_medals_predictions, decimals=0).astype(int)

    data_df = data_df.drop(labels=[200, 129, 194, 61, 203, 187], axis=0)
    data_df['Gold Predicted'] = gold_predictions
    data_df['Silver Predicted'] = silver_predictions
    data_df['Bronze Predicted'] = bronze_predictions

    # Not using total_medals_predictions as the below option gave slightly better results
    # top_df['Total Medals Predicted'] = total_medals_predictions

    data_df['Total Medals Predicted'] = data_df['Gold Predicted'] + data_df['Silver Predicted'] + \
                                        data_df['Bronze Predicted']
    return data_df