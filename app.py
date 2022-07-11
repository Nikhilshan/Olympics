import streamlit as st
import pandas as pd
import numpy as np
import preprocessor
import helper
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('athlete_events.csv')
region_df = pd.read_csv('noc_regions.csv')
olympic_df = pd.read_csv('Historical_Olympics_data.csv')
athle_df = pd.read_csv('athletes.csv')

df = preprocessor.preprocess(df,region_df)
olympic_df = preprocessor.his_preprocessor(olympic_df)


st.sidebar.title("Olympics Analysis")
st.sidebar.image('https://theaggie.org/wp-content/uploads/2021/05/olympicsfollowup_sp_CHRISTINA_LIU_AGGIE.png')
user_menu = st.sidebar.radio(
    'Select an Option',
    ('Home','Medal Tally','Overall Analysis','Country-wise Analysis','Athlete wise Analysis','Medal Prediction','Player Analysis')
)
if user_menu == 'Home':
    st.title('The Modern Olympic Games')
    st.image("Home Image.png",caption=None,width=None,use_column_width=None,clamp=False,channels="RGB",output_format="auto")

if user_menu == 'Medal Tally':
    st.sidebar.header("Medal Tally")
    years,country = helper.country_year_list(df)

    selected_year = st.sidebar.selectbox("Select Year",years)
    selected_country = st.sidebar.selectbox("Select Country", country)

    medal_tally = helper.fetch_medal_tally(df,selected_year,selected_country)
    if selected_year == 'Overall' and selected_country == 'Overall':
        st.title("Overall Tally")
    if selected_year != 'Overall' and selected_country == 'Overall':
        st.title("Medal Tally in " + str(selected_year) + " Olympics")
    if selected_year == 'Overall' and selected_country != 'Overall':
        st.title(selected_country + " overall performance")
    if selected_year != 'Overall' and selected_country != 'Overall':
        st.title(selected_country + " performance in " + str(selected_year) + " Olympics")
    st.table(medal_tally)

if user_menu == 'Overall Analysis':
    editions = df['Year'].unique().shape[0] - 1
    cities = df['City'].unique().shape[0]
    sports = df['Sport'].unique().shape[0]
    events = df['Event'].unique().shape[0]
    athletes = df['Name'].unique().shape[0]
    nations = df['region'].unique().shape[0]

    st.title("Top Statistics")
    col1,col2,col3 = st.columns(3)
    with col1:
        st.header("Editions")
        st.title(editions)
    with col2:
        st.header("Hosts")
        st.title(cities)
    with col3:
        st.header("Sports")
        st.title(sports)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Events")
        st.title(events)
    with col2:
        st.header("Nations")
        st.title(nations)
    with col3:
        st.header("Athletes")
        st.title(athletes)

    nations_over_time = helper.data_over_time(df,'region')
    fig = px.line(nations_over_time, x="Edition", y="region")
    st.title("Participating Nations over the years")
    st.plotly_chart(fig)

    events_over_time = helper.data_over_time(df, 'Event')
    fig = px.line(events_over_time, x="Edition", y="Event")
    st.title("Events over the years")
    st.plotly_chart(fig)

    athlete_over_time = helper.data_over_time(df, 'Name')
    fig = px.line(athlete_over_time, x="Edition", y="Name")
    st.title("Athletes over the years")
    st.plotly_chart(fig)

    st.title("No. of Events over time(Every Sport)")
    fig,ax = plt.subplots(figsize=(20,20))
    x = df.drop_duplicates(['Year', 'Sport', 'Event'])
    ax = sns.heatmap(x.pivot_table(index='Sport', columns='Year', values='Event', aggfunc='count').fillna(0).astype('int'),
                annot=True)
    st.pyplot(fig)

    st.title("Most successful Athletes")
    sport_list = df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0,'Overall')

    selected_sport = st.selectbox('Select a Sport',sport_list)
    x = helper.most_successful(df,selected_sport)
    st.table(x)

if user_menu == 'Country-wise Analysis':

    st.sidebar.title('Country-wise Analysis')

    country_list = df['region'].dropna().unique().tolist()
    country_list.sort()

    selected_country = st.sidebar.selectbox('Select a Country',country_list)

    country_df = helper.yearwise_medal_tally(df,selected_country)
    fig = px.line(country_df, x="Year", y="Medal")
    st.title(selected_country + " Medal Tally over the years")
    st.plotly_chart(fig)

    st.title(selected_country + " excels in the following sports")
    pt = helper.country_event_heatmap(df,selected_country)
    fig, ax = plt.subplots(figsize=(20, 20))
    ax = sns.heatmap(pt,annot=True)
    st.pyplot(fig)

    st.title("Top 10 athletes of " + selected_country)
    top10_df = helper.most_successful_countrywise(df,selected_country)
    st.table(top10_df)

if user_menu == 'Athlete wise Analysis':
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])

    x1 = athlete_df['Age'].dropna()
    x2 = athlete_df[athlete_df['Medal'] == 'Gold']['Age'].dropna()
    x3 = athlete_df[athlete_df['Medal'] == 'Silver']['Age'].dropna()
    x4 = athlete_df[athlete_df['Medal'] == 'Bronze']['Age'].dropna()

    fig = ff.create_distplot([x1, x2, x3, x4], ['Overall Age', 'Gold Medalist', 'Silver Medalist', 'Bronze Medalist'],show_hist=False, show_rug=False)
    fig.update_layout(autosize=False,width=1000,height=600)
    st.title("Distribution of Age")
    st.plotly_chart(fig)

    x = []
    name = []
    famous_sports = ['Basketball', 'Judo', 'Football', 'Tug-Of-War', 'Athletics',
                     'Swimming', 'Badminton', 'Sailing', 'Gymnastics',
                     'Art Competitions', 'Handball', 'Weightlifting', 'Wrestling',
                     'Water Polo', 'Hockey', 'Rowing', 'Fencing',
                     'Shooting', 'Boxing', 'Taekwondo', 'Cycling', 'Diving', 'Canoeing',
                     'Tennis', 'Golf', 'Softball', 'Archery',
                     'Volleyball', 'Synchronized Swimming', 'Table Tennis', 'Baseball',
                     'Rhythmic Gymnastics', 'Rugby Sevens',
                     'Beach Volleyball', 'Triathlon', 'Rugby', 'Polo', 'Ice Hockey']
    for sport in famous_sports:
        temp_df = athlete_df[athlete_df['Sport'] == sport]
        x.append(temp_df[temp_df['Medal'] == 'Gold']['Age'].dropna())
        name.append(sport)

    fig = ff.create_distplot(x, name, show_hist=False, show_rug=False)
    fig.update_layout(autosize=False, width=1000, height=600)
    st.title("Distribution of Age wrt Sports(Gold Medalist)")
    st.plotly_chart(fig)

    sport_list = df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0, 'Overall')

    st.title('Height Vs Weight')
    selected_sport = st.selectbox('Select a Sport', sport_list)
    temp_df = helper.weight_v_height(df,selected_sport)
    fig,ax = plt.subplots()
    ax = sns.scatterplot(temp_df['Weight'],temp_df['Height'],hue=temp_df['Medal'],style=temp_df['Sex'],s=60)
    st.pyplot(fig)

    st.title("Men Vs Women Participation Over the Years")
    final = helper.men_vs_women(df)
    fig = px.line(final, x="Year", y=["Male", "Female"])
    fig.update_layout(autosize=False, width=1000, height=600)
    st.plotly_chart(fig)
if user_menu == 'Medal Prediction':
    del olympic_df['CountryId']
    st.title('Medal Prediction Table')
    con = np.unique(olympic_df['Country'].dropna().values).tolist()
    con.sort()
    con.insert(0, 'Overall')
    selected_co = st.sidebar.selectbox("Select Country", con)
    if selected_co == 'Overall':
        st.table(olympic_df)
    else:
        st.table(olympic_df[olympic_df['Country'] == selected_co])
if user_menu == 'Player Analysis':
    st.sidebar.header("Player Analysis")
    del athle_df['dob'], athle_df['id'], athle_df['name']
    athle_df['nationality'] = athle_df['nationality'].astype(str)
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(athle_df.iloc[:, 2:4])
    athle_df.iloc[:, 2:4] = imputer.transform(athle_df.iloc[:, 2:4])
    x = athle_df.iloc[:, 0:5].values.transpose()
    y = athle_df.iloc[:, [5, 6, 7]].values
    le = LabelEncoder()
    x[0] = le.fit_transform(x[0])
    x[4] = le.fit_transform(x[4])
    x[1] = le.fit_transform(x[1])
    gender = athle_df['sex'].values
    nationality = athle_df['nationality'].values
    sport = athle_df['sport'].values
    cat_gen = dict(zip(gender, x[1]))
    cat_nation = dict(zip(nationality, x[0]))
    cat_sport = dict(zip(sport, x[4]))
    full_data = [cat_nation, cat_gen, cat_sport]
    x = x.transpose()
    ranFor = RandomForestRegressor(n_estimators=245, random_state=40, max_depth=14, min_samples_split=10)
    ranFor.fit(x, y)


    def inputs_conv(country1, gender1, height1, weight1, sport1):
        cnt = full_data[0].get(country1.upper())
        gen = full_data[1].get(gender1.lower())
        spt = full_data[2].get(sport1.lower())
        hgt = height1
        wt = weight1
        list_all = [cnt, gen, hgt, wt, spt]
        pred = ranFor.predict([list_all])


        # Gold medal prediction
        if pred[0][0] < 0.036:
            gold = 0
        elif pred[0][0]> 0.036 and pred[0][0] < 0.12:
            gold = 1
        elif pred[0][0]>0.12 and pred[0][0] < 0.9:
            gold = 2
        elif pred[0][0] > 0.9:
            gold = '2+'

        # Silver medal prediction
        if pred[0][1] < 0.3:
            silver = 0
        elif 0.3 < pred[0][1] < 0.5:
            silver = 1
        elif pred[0][1] > 0.5:
            silver = '1+'

        # Bronze medal prediction
        if pred[0][2] < 0.21:
            bronze = 0
        elif 0.21 < pred[0][2] < 4.5:
            bronze = 1
        elif pred[0][2] > 4.5:
            bronze = '1+'

        predict = f'Gold : {gold} [{round(pred[0][0], 2)}%] \nSilver: {silver} [{round(pred[0][1], 2)}%]\nBronze: {bronze} [{round(pred[0][2], 2)}%]\n'
        return predict, pred
    noc = athle_df['nationality'].unique().tolist()
    selected_noc = st.selectbox("Select Country", noc)
    selected_gender = st.selectbox("Choose Gender", ('male', 'female'))
    selected_height = st.number_input("Height(in meters)", min_value=1.00, max_value=3.00, step=0.01)
    selected_weight = st.number_input("Weight(in kgs)", min_value=20, max_value=200, step=1)
    ss = athle_df['sport'].unique().tolist()
    selected_sport = st.selectbox("Select the sport", ss)
    p = inputs_conv(selected_noc, selected_gender, selected_height, selected_weight, selected_sport)
    st.text(p[0])
    no = 1 - (p[1][0][0] + p[1][0][1] + p[1][0][2])
    values = [p[1][0][0], p[1][0][1], p[1][0][2], no]
    exp_labels = ['Gold', 'Silver', 'Bronze', 'No medal']
    colors = ['yellow', 'grey', 'orange', 'red']
    plt.pie(values, labels=exp_labels, autopct='%0.1f%%', explode=[0.2, 0.2, 0.2, 0], colors=colors)
    st.title('Winning Chances')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

