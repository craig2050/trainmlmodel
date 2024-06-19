
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import seaborn as sns

st.set_page_config(page_title = "My Webpage", page_icon = ":taga:", layout = 'wide') #, layout = 'wide'



def create_histogram_bins(data, num_bins):
    # Calculate bin edges
    bin_edges = []
    min_val = round(min(data))
    max_val = round(max(data))
    bin_width = round( (max_val - min_val) / num_bins )
    
    for i in range(num_bins + 1):
        bin_edges.append(min_val + i * bin_width)
    
    # Create bins
    bins = []
    for i in range(len(bin_edges) - 1):
        bins.append((bin_edges[i], bin_edges[i+1]))
    
    # Calculate frequency per bin
    frequencies = []
    for bin in bins:
        frequency = sum(1 for d in data if bin[0] <= d < bin[1])
        frequencies.append(frequency)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Bin': [f'{bin[0]} - {bin[1]}' for bin in bins],
        'Frequency': frequencies
    })
    
    return df

header = st.container()
dataset = st.container()

visuals = st.container()
visuals_1 = st.container()
features = st.container()

model_training = st.container()


st.write('----')
with header:
	st.write('----')
	st.title('Train your own ML model')
	st.subheader('Predicting used tesla car prices')
	st.write('''In this project start with exploring the datset through some visualizations.
    The underlining goal for the project is to allow the user to setup a custom Tree based ML model,
    and adjust the hyperparameters for the model to achive a accuracy of 90 percent ''')


df = pd.read_csv('tslaUsedCarsInventoryDataSet.csv')
num_columns = df.select_dtypes(include=['int','float']).columns.to_list()
cat_columns = df.select_dtypes(include=['O']).columns.to_list()

with dataset:
	st.write('----')
	st.header('Tesla Used Cars Dataset')
	st.markdown("I have used a dataset available on [Kaggle](https://www.kaggle.com/datasets/aravindrajpalepu/tesla-used-cars)")
	st.write('The data lists used tesla cars available in the US, the data itself was scaraped from tesla used car listings')
	st.write('Lets take a look at the the datset')
	df.dropna(inplace=True)
	st.write(df.head(3))
	st.write(f'The dataset had {df.shape[0]} rows and {df.shape[1]} columns')
	st.write(f'These columns are in numerical format : {num_columns}')
	st.write(f'These columns are in categorical format : {cat_columns}')



# with visuals:
# 	st.write('----')
# 	st.write('----')
# 	st.title('Vizualizations')

# 	left_column, right_column = st.columns(2)
	
# 	# Left Column representation

# 	#Distribution by model
# 	left_column.subheader('Popular Model Type')
# 	model_name_dist = pd.DataFrame(df['model'].value_counts())
# 	left_column.bar_chart(model_name_dist, width=50, color="#ffaa00")
# 	#Distribution by price
# 	right_column.subheader('Price distribution')
# 	price_data = df['price']
# 	price_data = create_histogram_bins(price_data, num_bins = 50)
# 	right_column.bar_chart(price_data, x = 'Bin', width=0, color="#ffaa00")
# 	#Distribution by state
# 	st.write('----')
# 	st.subheader("State wise distribution of tesla's ")
# 	state_dist = pd.DataFrame(df['state'].value_counts())
# 	st.write("looks like state of california has signigicantly higher number of tesla's")
# 	st.bar_chart(state_dist, width=50, color="#ffaa00")


with visuals:
	st.write('----')
	st.write('----')
	st.title('Price Distribution by Model')
	selected_model = st.selectbox('Select Model', ['All', 'Model S', 'Model 3', 'Model X'])

	if selected_model != 'All':
		df = df[df['model'] == selected_model]
	else:
		pass

	left_column, right_column = st.columns(2)
	
	
	# Left Column representation

	#Distribution by model

	left_column.subheader('Price Distribution by Model')
	fig1, ax1 = plt.subplots()
	sns.boxplot(x='model', y='price', data=df, ax=ax1)
	# ax1.set_title('Price Distribution by Model')
	model_name_dist = pd.DataFrame(df['model'].value_counts())
	left_column.pyplot(fig1)
	left_column.write('Model S is heavily skewed to the right, and this may be owing to trims and vehicle packs')


	#Distribution by model
	right_column.subheader('Histogram plot')
	fig1, ax1 = plt.subplots()
	sns.histplot(x='price', data=df, ax=ax1)
	# ax1.set_title('Price Distribution by Model')
	model_name_dist = pd.DataFrame(df['model'].value_counts())
	right_column.pyplot(fig1)
	right_column.write('Plots show data is skewed')

	


	st.write('----')
	st.subheader("State wise distribution of tesla's ")
	state_dist = pd.DataFrame(df['state'].value_counts().reset_index())
	fig1,ax1 = plt.subplots(figsize = (10,2.5))
	sns.barplot(x = 'state', y = 'count', data = state_dist, ax = ax1 , color = 'navy')
	ax1.set_ylabel('Number of vehicles')
	ax1.set_xlabel('State')
	st.write("looks like state of california has signigicantly higher number of tesla's")
	st.pyplot(fig1)

if 'oob_scores' not in st.session_state:
    st.session_state.oob_scores = []

with model_training:
	st.write('----')
	st.write('----')
	st.header('Lets Train a ML Model to predict price of a used tesla')
	st.write('''
		You will able to choose the hyper parameters for training a Random forest ML model, The goal is to get a score above **90%** :goal_net: !!!

		''')

	dis_col, sel_col  = st.columns(2)
	st.write('----')

	dis_col.write('Adjust the hyper parameters on the right and train the ML model using the "Run Model" button below')

	input_features = sel_col.multiselect('Which features to be used as an input ?', options = list((set(df.columns).difference(set(['price'])))), placeholder = 'Select features')
	max_depth = sel_col.slider('what should be the max depth of the model', min_value = 1, max_value = 10, step = 1, value = 2)
	min_samples_leaf = sel_col.slider('Minimum number of samples at leaf node', min_value = 1, max_value = 15, step = 1, value = 4)
	max_features = sel_col.slider('Maximum features to be selected for each tree', min_value = 1, max_value = 10, step = 1, value = 1)
	n_estimators = sel_col.selectbox('how many trees do you want ?', options = [30,50,100,200,400])
	
	
	rndfor = RandomForestRegressor(max_depth = max_depth, n_estimators = n_estimators)
	oob_score = 0


	X = df[input_features]
	y = df[['price']]

	categorical_cols = [col for col in X.columns if X[col].dtype == 'object']

	preprocessor = ColumnTransformer(
    transformers=	[
        			('cat', OneHotEncoder(), categorical_cols)  # Apply OneHotEncoder to categorical columns
    				],
    					remainder='passthrough'  # Keep the remaining numeric columns unchanged
					)
	rndfor = RandomForestRegressor(max_depth=max_depth, min_samples_leaf = min_samples_leaf,max_features = max_features,
									 n_estimators=n_estimators, oob_score = True)

	# Create the pipeline by combining the preprocessor and the regressor
	pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', rndfor)])
	


	if dis_col.button('Run Model', help = 'Update hyperparameters before training'):

		if len(input_features) >= 1:

			# Fit the pipeline
			pipeline.fit(X, y)

			# Make predictions
			predictions = pipeline.predict(X)

			oob_score = round(pipeline.named_steps['regressor'].oob_score_, 3)
			st.session_state.oob_scores.append(oob_score)
			dis_col.markdown(f'<p style="font-size:32px;">Model Performance: {round((oob_score*100),2)}%</p>', unsafe_allow_html=True)
		else:
			# dis_col.markdown(f'<p style="font-size:32px;">Select some features for prediction</p>', unsafe_allow_html=True)
			dis_col.warning('First select some features for prediction',icon="🚨")
		
	df_scores = pd.DataFrame(st.session_state.oob_scores, columns = ['score'])
	fig1, ax1 = plt.subplots(figsize = (5,1.5))
	sns.lineplot(x=np.arange(1,df_scores.shape[0]+1), y='score', data=df_scores, ax=ax1)
	ax1.set_title('Model Scores graph')
	dis_col.pyplot(fig1)

	if oob_score > 0.90:
		st.success('Awesome!', icon="✅")
		st.balloons()


	# disp_col.subheader('Mean absolute error')
	# disp_col.subheader('Mean squared error')
	# disp_col.subheader('R squared score of the model')
	# disp_col.subheader('Importtance of input features')
