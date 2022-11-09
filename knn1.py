import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('./KNNAlgorithmDataset.csv')
X = dataset.iloc[:, 0:4].values
dataset1 = pd.DataFrame(X)
dataset1.columns =['id', 'diagnosis', 'radius_mean', 'texture_mean']

le = LabelEncoder()
y = le.fit_transform(dataset1['diagnosis'])

x = dataset1.iloc[:,2:4].values

def normalize(point):
    minimum = min(point)
    maximum = max(point)
    normalized_val = [(value - minimum)/(maximum - minimum) for value in 
    point]                               
    return normalized_val

def euclidean_distance(a , b):
    x_distance = (a[0] - b[0])**2
    y_distance = (a[1] - b[1])**2
    return (x_distance + y_distance)**0.5

def get_label(neighbours, y):
    zero_count , one_count = 0,0
    for element in neighbours:
      if y[element[1]] == 0:
         zero_count +=1
      elif y[element[1]] == 1:
         one_count +=1
    if zero_count == one_count:
         return y[neighbours[0][1]]
    return 1 if one_count > zero_count else 0
    
def find_nearest(x , y , input , k):
    distances = []
    for id,element in enumerate(x):
        distances.append([euclidean_distance(input , element),id])
    distances = sorted(distances)
    predicted_label = get_label(distances[0:k] , y)
    print(predicted_label)
    return predicted_label, distances[0:k]
    
input = (0.72,0)

x[:,0] = normalize(x[:,0])
x[:,1] = normalize(x[:,1])

df = pd.DataFrame(x, columns=['radius_mean', 'texture_mean'])
df['labels'] = y

st.dataframe(df)
fig = px.scatter(df, x = 'radius_mean' , y='texture_mean', symbol='labels',symbol_map={'0':'square-dot' , '1':'circle'})
fig.add_trace(
    go.Scatter(x= [input[0]], y=[input[1]], name = "Point to Classify", )
)
st.plotly_chart(fig)

#Finding Nearest Neighbours
predicted_label , nearest_neighbours = find_nearest(x ,y , input ,k=5)

st.title('Prediction')
st.subheader('Predicted Label : {}'.format(predicted_label))

predicted_label , nearest_neighbours= find_nearest(x ,y , input ,5)

nearest_neighbours = [[neighbour[1],x[neighbour[1],0],x[neighbour[1],1],neighbour[0],y[neighbour[1]]] for neighbour in nearest_neighbours]

nearest_neighbours = pd.DataFrame(nearest_neighbours , columns = ['id','Feature1','Feature2','Distance','Label'])

print(nearest_neighbours)