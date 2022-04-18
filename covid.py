

# import libraries
from turtle import showturtle
import matplotlib.pyplot as plt
import plotly.express as px
from pyparsing import line_end
import streamlit as st
import pandas as pd
import numpy as np


# def gt_total_df(df):
#     t_df = pd.DataFrame({ 'Status': ['Confirmed', 'Recovered','Death','Active'],
#                             'Number of cases':(df.iloc[0]['confirmed_cases'],
#                             df.iloc[0]['active_cases'],
#                             df.iloc[0]['recovered_cases'],df.iloc[0],['death_cases'])})
#     return t_df


st.title ( "Pandemic COVID-19 Analysis " )
st.write ("COVID-19 is the latest infectious disease effect the whole world. Let's visualize this change over the months in spread with Python and Machine Learning.  " )


# download world covid dataset 
df = pd.read_csv('covid-data.csv')

# dependent_var = ['total_deaths','total_deaths_per_million','total_cases_per_million','icu_patients_per_million',
#                 'people_vaccinated_per_hundred','total_vaccinations','hosp_patients_per_million']

# independent_var = ['gdp_per_capita','population','stringency_index','population', 
#                     'population_density','median_age','aged_65_older','aged_70_older','gdp_per_capita','extreme_poverty',
#                     'cardiovasc_death_rate','diabetes_prevalence','female_smokers', 'male_smokers','handwashing_facilities',
#                     'hospital_beds_per_thousand','life_expectancy','continent','location']


# df.drop( [ 'excess_mortality_cumulative_absolute', 'excess_mortality_cumulative', 'excess_mortality', 'excess_mortality_cumulative_per_million' ],  axis = 1 , inplace = True )

# Select variable to analyse
df1 = df[['date', 'location' ,'total_cases' ,'new_cases', 'total_deaths' ,'new_deaths', 'icu_patients' , 'total_tests', 'new_tests', 'population', 'total_tests_per_thousand' ]]

# Select the column names to plot.
st.markdown(" **Select country to anlyse statistics of COVID-19**")
country_sel = st.selectbox ("Country :" , df['location'].unique() )

# Append list of selected country to the bellow famous countries list 
df_state = [ 'Pakistan' 'Australia', 'France' , 'Germany' , 'India', 'China', 'United Kingdom' , 'United States' , 'Italy' , 'Israel' ,
             'Italy' , 'Brazil', 'China' , 'Canada']
df_state.append(str(country_sel))

# Now select dataframe values of desired country rows in csv dataset 
df_sel_state = df1[df1['location'].isin(df_state)]
df_pak = df1[df1['location'] == 'Pakistan']
#  Fill NAN value with 0
df1.fillna( 0 )

print(df.shape)

# print('Date span:',df.date.min(),df.date.max())
st.markdown( "*choose country to analysis COVID-19 Cases*")


# df_state = np.concatenate(df_state , country_sel)

st.write ( df[df['location'] == country_sel].sample(100) )
st.markdown("---")


# stat_lbl = st.sidebar.selectbox('Covid-19 status', (df1.columns) )
st.sidebar.header('Choose the graph to plot.')
plot_type = st.sidebar.radio(
        "Select a chart type",
        ('Bar_plot', 'Pi-plot', 'Line-plot'))
        

st.sidebar.header('Status Select')
status_select = st.sidebar.radio(
        "Status Select",
        ('Confirmed_Cases', 'Test_Cases', 'Death_Cases', 'Recovered_Cases'))
# status_ttl = gt_total_df(df)
# st.write ( status_ttl )

if plot_type == 'Bar_plot':
    st.title ( "Bar Chart")
    if status_select == 'Confirmed_Cases':
        # st.header('Total Confirmed Cases')
        fig =  px.bar( df_sel_state ,  x= df_sel_state['location'], y = df_sel_state['total_cases'], color= df_sel_state['location'],text_auto='.2s',  title = 'Total Confirmed Cases' )
    elif status_select == 'Test_Cases':
        # st.header('Total Test_Cases')
        fig =  px.bar( df_sel_state ,  x= df_sel_state['location'], y = df_sel_state['total_tests'], color= df_sel_state['location'] ,text_auto='.2s', title = 'Total Test Cases' )
    elif status_select == 'Death_Cases':
        # st.header('Total Death_Cases')
        fig =  px.bar( df_sel_state ,  x= df_sel_state['location'], y = df_sel_state['total_deaths'], color= df_sel_state['location'] , text_auto='.2s', title= 'Total Death Cases')
    else: 
        df_sel_state['recovery'] =  df_sel_state['total_cases'] - df_sel_state['total_deaths']
        fig =  px.bar( df_sel_state ,  x= df_sel_state['location'], y = df_sel_state['recovery'] , color= df_sel_state['location'] , text_auto='.2s', title = 'Total Recovered Cases')
    
    fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False, opacity=0.8)
    fig.update_layout( autosize=True,xaxis_title='Country',yaxis_title = 'Number of Cases',showlegend=True,
        plot_bgcolor="white", margin=dict(t=10,l=10,b=10,r=10)   )

    st.plotly_chart(fig)

elif plot_type == 'Pi-plot':
    st.title( "Pi Chart")
    if status_select == 'Confirmed_Cases':
        st.header('Total Confirmed Cases')
        fig = px.pie ( df_sel_state, values= df_sel_state['new_cases'] , names= df_sel_state['location']  )
    elif status_select == 'Test_Cases':
        st.header('Total Test_Cases')
        fig = px.pie ( df_sel_state, values= df_sel_state['new_tests'] , names=  df_sel_state['location'])
    elif status_select == 'Death_Cases':
        st.header('Total Death_Cases')
        fig = px.pie ( df_sel_state, values= df_sel_state['new_deaths'] , names=  df_sel_state['location'] )
    else: 
        st.header('Total Recovered Cases')
        fig = px.pie ( df_sel_state, values= df_sel_state['new_cases'] - df_sel_state['new_deaths'] , names=  df_sel_state['location'] )
    st.plotly_chart(fig)


elif plot_type == 'Line-plot':
    st.title( "Line Chart")
    if status_select == 'Confirmed_Cases':
        if st.checkbox("Log", value=False):        
            df_sel_state['total_cases'] = np.log10( df_sel_state['total_cases'] )
            fig = px.line( df_sel_state, x = df_sel_state['date'], y= df_sel_state['total_cases'] , color= df_sel_state['location'])        
        else:        
            fig = px.line( df_sel_state, x = df_sel_state['date'], y= df_sel_state['total_cases'] , color= df_sel_state['location'] )
    
    elif  status_select == 'Test_Cases':
        st.header('Death Cases')
        if st.checkbox("Log", value=False):        
            df_sel_state['total_tests'] = np.log10( df_sel_state['total_tests'] )
            fig = px.line( df_sel_state, x = df_sel_state['date'], y= df_sel_state['total_tests'] , color= df_sel_state['location'])        
        else:        
            fig = px.line( df_sel_state, x = df_sel_state['date'], y= df_sel_state['total_tests'] , color= df_sel_state['location'] )
    
    elif  status_select == 'Death_Cases':
        st.header('Death Cases')
        if st.checkbox("Log", value=False):        
            df_sel_state['total_deaths'] = np.log10( df_sel_state['total_deaths'] )
            fig = px.line( df_sel_state, x = df_sel_state['date'], y= df_sel_state['total_deaths'] , color= df_sel_state['location'])        
        else:        
            fig = px.line( df_sel_state, x = df_sel_state['date'], y= df_sel_state['total_deaths'] , color= df_sel_state['location'] )
    
    else :
        st.header('Total Recovered Cases')
        if st.checkbox("Log", value=False):        
            fig = px.line( df_sel_state, x = df_sel_state['date'], y= np.log10(df_sel_state['total_cases'] - df_sel_state['total_deaths']) , color= df_sel_state['location'])        
        else:        
            fig = px.line( df_sel_state, x = df_sel_state['date'], y= df_sel_state['total_cases'] - df_sel_state['total_deaths'] , color= df_sel_state['location'] )
    

    
    fig.update_traces(line_width=2, mode='lines' )
    fig.update_layout(
        xaxis_title='Date',
        showlegend=True,
        plot_bgcolor="white",
        margin=dict(t=10,l=10,b=10,r=10)
    )
    st.plotly_chart(fig)

# col1 ,col2 = st.columns(2)
# #Bar Chart
# st.header('Select the column names to plot.')
# bx_lbl = col1.selectbox('x axis: ', df1.columns)
# by_lbl = col1.selectbox('y axis: ', df1.drop(columns=[bx_lbl]).columns)
# bz_lbl = col1.selectbox('z axis: ', df1.drop(columns=[bx_lbl, by_lbl]).columns)


# # plot
# fig = px.bar(df1, x='date', y = by_lbl)
# col2.write(fig)

# st.markdown("---")

# st.markdown ( " **Pi Chart**")

# col3 ,col4 = st.columns(2)
# #Bar Chart
# st.header('Select the column names to plot.')
# # px_lbl = col3.selectbox('x Axis: ', df1.columns)
# # py_lbl = col3.selectbox('y Axis: ', df1.drop(columns=[px_lbl]).columns)
# # z_lbl = col3.sidebar.selectbox('z axis: ', df1.drop(columns=[x_lbl, y_lbl]).columns)

# st.markdown("----")
# st.markdown ( " **PI Chart**")
# cases = df1['new_cases'].sum()
# deceased = df1['new_deaths'].sum()

# labels =  ["Confirmed", "Deceased"]
# values = [cases, deceased]


# fig = px.pie ( df1 , values=values, 
#              names=labels, 
#              title='Confirmed vs Deceased ')

# col4.write(fig)

# st.markdown( '---')

# # Let’s calculate the death rate of Covid-19:


# # df1['death_rate'] = (df1["new_cases"].sum() / df1["total_cases"].sum()) * 100
# # df1['death_rate'] =  deceased/ cases* 100
# # # print(df_sel['death_rate'].head())

# # plt.figure(figsize=(12,8))
# # # plt.hist(100 * np.array(df_sel['death_rate']), bins=np.arange(1))
# # plt.hist( df1['death_rate']  )
# # p.xlabel('Death Rate (%)', fontsize=16)
# # plt.ylabel('Number of cases', fontsize=16)
# # plt.title('Histogram of Death Rates', fontsize=18)
# # # plt.show()
# # st.write( plt)



# df1['Num Tests per Positive Case'] = df1['Total Tests'] / df1['Total Cases']

# min_number_of_cases = 1000
# greatly_affected_df = df1.loc[df1['Total Cases'] > min_number_of_cases,:]

# x_axis_limit = 80

# death_rate_percent = 100 * np.array(greatly_affected_df['Case Fatality Ratio'])
# num_test_per_positive = np.array(greatly_affected_df['Num Tests per Positive Case'])
# num_test_per_positive[num_test_per_positive > x_axis_limit] = x_axis_limit
# total_num_deaths = np.array(greatly_affected_df['Total Deaths'])
# population = np.array(greatly_affected_df['Population'])

# # plot on top of the figure the names of the
# #countries_to_display = greatly_affected_df['Country'].unique().tolist()
# countries_to_display = ['USA', 'Russia', 'Spain', 'Brazil', 'UK', 'Italy', 'France', 
#                         'Germany', 'India', 'Canada', 'Belgium', 'Mexico', 'Netherlands', 
#                         'Sweden', 'Portugal', 'UAE', 'Poland', 'Indonesia', 'Romania', 
#                         'Israel','Thailand','Kyrgyzstan','El Salvador', 'S. Korea', 
#                         'Denmark', 'Serbia', 'Norway', 'Algeria', 'Bahrain','Slovenia',
#                         'Greece','Cuba','Hong Kong','Lithuania', 'Australia', 'Morocco', 
#                         'Malaysia', 'Nigeria', 'Moldova', 'Ghana', 'Armenia', 'Bolivia', 
#                         'Iraq', 'Hungary', 'Cameroon', 'Azerbaijan']

# for country_name in countries_to_display:
#     country_index = greatly_affected_df.index[greatly_affected_df['Country'] == country_name]
#     fig = plt.text(x=num_test_per_positive[country_index] + 0.5,
#              y=death_rate_percent[country_index] + 0.2,
#              s=country_name, fontsize=10)
# st.write(fig)








