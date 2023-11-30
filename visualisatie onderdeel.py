import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import dash
import wordcloud
from dash import dcc, html
import dash.dependencies as dd
from dash.dependencies import Input, Output
from dash import dash_table
from typing import Counter
from PIL import Image
import warnings
import plotly.express as px  # Voeg import toe
import plotly.graph_objects as go  # Voeg import toe
import MongoDatabase_onderdeel as mbd
warnings.filterwarnings('ignore')
import flask
import glob
import os  # Voeg import toe

# Ophalen van gegevens uit de database

hotel_reviews = mbd.laad_data("kaggle_dataset")

def extract_city_and_country(address):
    parts = address.split()
    country = parts[-1]
    city = parts[-2]
    return city, country
hotel_reviews[['city', 'country']] = hotel_reviews['Hotel_Address'].apply(lambda x: pd.Series(extract_city_and_country(x)))

df = hotel_reviews[['Hotel_Name', 'Average_Score', 'Total_Number_of_Reviews', 'country', 'city', 'Hotel_Address','Additional_Number_of_Scoring','lat','lng']]

nationalities = pd.DataFrame.from_dict(Counter(hotel_reviews['Reviewer_Nationality']), orient='index', columns=['country']).reset_index()

image_directory = r"C:\Users\thijs\Desktop\assignment 2\wordclouds"

list_of_images = [os.path.join(image_directory, file) for file in os.listdir(image_directory) if file.endswith(".png")]
static_image_route = '/static/'

logo_link = (r"C:\Users\thijs\Desktop\assignment 2\fotos\hva_logo")
app = dash.Dash(__name__)

# Voeg een div toe voor het tonen van details over een geselecteerd hotel
selected_hotel_details = html.Div(id='selected-hotel-details', style={'padding': '20px'})

mapa = px.scatter_mapbox(df, lat="lat", lon="lng", color="Average_Score", size="Total_Number_of_Reviews",
                  color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10, hover_name=df.Hotel_Name)

mapa.update_layout(
    mapbox_style="open-street-map",
    title='Hotels review score kaart',
    autosize=True,
    hovermode='closest',
    showlegend=False,
)
data_scatter = mbd.aggregate("kaggle_dataset")

triple = px.scatter(data_scatter,x="Additional_Number_of_Scoring", y="Average_Score", size="Total_Number_of_Reviews", color="Hotel_Name",
           hover_name="Hotel_Name", size_max=60, width=920, title = 'Hotels scores en aantal reviews')

# Pie chart
pie = px.pie(nationalities.query("country >= 5000"), values='country', names='index', title='Nationaliteiten van de reviewers')


logo_link = 'https://zakelijkschrijven.nl/wp-content/uploads/2021/01/HvA-logo.png'
  
# data_top_num_reviews = df.nlargest(10, 'Total_Number_of_Reviews')
# fig1 = px.bar(data_top_num_reviews, x='Hotel_Name', y='Total_Number_of_Reviews', color='Hotel_Name')                 
list_names = ['Name', 'Average score', 'Number of reviews', 'Country', 'City', 'Adress', 'Additional score', 'lat', 'long']
cols=[
            {"name": i, "id": i, "deletable": True, "selectable": True} for i in df.columns
        ]
cols[0]["name"] = "Naam"
cols[1]["name"] = "Gemiddelde score"
cols[2]["name"] = "Aantal reviews"
cols[3]["name"] = "Land"
cols[4]["name"] = "Stad"
cols[5]["name"] = "Adres"
cols[6]["name"] = "Aanvullende score"
cols[7]["name"] = "lat"
cols[8]["name"] = "long"


opts =[{'label': i, 'value': i} for i in list_of_images]
opts[0]["label"] = "cloud_negatief"
opts[1]["label"] = "cloud_positief"
opts[2]["label"] = "totaal_cloud"
    
app.layout = html.Div([
    html.Img(src=logo_link, width="500", height="250"),
    html.H1('Dashboard van hotel reviews', style={'font-size': '32px', 'color': 'navy'}),

    # Voeg filters toe
    dcc.Dropdown(
        id='country-filter',
        options=[{'label': country, 'value': country} for country in df['country'].unique()],
        multi=True,
        placeholder='Selecteer land'
    ),
    dcc.RangeSlider(
        id='score-filter',
        min=0,
        max=10,
        step=0.1,
        marks={i: str(i) for i in range(0, 11)},
        value=[0, 10],
        allowCross=False,
        pushable=0.1
    ),

    html.Div(className='row', children=[
        html.Div(className='six columns', children=[
            dash_table.DataTable(
                id='datatable-interactivity',
                columns=cols,
                data=df.to_dict('records'),
                style_table={'overflowX': 'auto'},
                editable=True,
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                column_selectable="single",
                row_selectable="multi",
                row_deletable=True,
                selected_columns=[],
                selected_rows=[],
                page_action="native",
                page_current=0,
                page_size=10,
                style_cell={
                    'height': 'auto',
                    'minWidth': '180px',
                    'width': '90px',
                    'maxWidth': '90px',
                    'textAlign': 'left'
                },
                style_cell_conditional=[
                    {
                        'if': {'column_id': 'Hotel_Name'},
                        'textAlign': 'left'
                    }
                ]
            ),
        ]),
        html.Div(className='six columns', children=[
            html.Div(children=[
                dcc.Graph(
                    id='triple-graph',
                    figure=triple,
                ),
                dcc.Graph(
                    figure=pie,
                )
            ]),
        ]),
    ]),

    html.Div(style={'text-align': 'center', 'font-size': 22}, children=[
        dcc.Graph(
            id='example-map',
            figure=mapa,
        ),
    ]),

    html.Div(style={'margin-top': '20px'}, children=[
        dcc.Dropdown(
            id='image-dropdown',
            options=opts,
            value=list_of_images[2],
        ),
        html.Img(id='image'),
    ]),
    selected_hotel_details,
], style={'text-align': 'center', 'font-size': 22})



# Callback voor het bijwerken van de geselecteerde hotelgegevens
@app.callback(
    Output('selected-hotel-details', 'children'),
    Input('datatable-interactivity', 'selected_rows'),
)
def update_selected_hotel_details(selected_rows):
    if selected_rows:
        selected_hotel = df.iloc[selected_rows[0]]  # Neem het eerste geselecteerde hotel
        # Stel de lay-out in voor het weergeven van gedetailleerde informatie over het hotel
        return html.Div([
            html.H2(selected_hotel['Hotel_Name']),
            html.P(f'Average Score: {selected_hotel["Average_Score"]:.2f}'),
            html.P(f'Total Number of Reviews: {selected_hotel["Total_Number_of_Reviews"]}')
            # Voeg hier andere details van het hotel toe
        ])
    else:
        return html.Div()  # Leeg div als er geen hotel is geselecteerd

# Callback voor het filteren van gegevens op basis van land en scores
@app.callback(
    Output('datatable-interactivity', 'data'),
    Input('country-filter', 'value'),
    Input('score-filter', 'value'),
)
def update_filtered_data(selected_countries, score_range):
    filtered_data = df
    if selected_countries:
        filtered_data = filtered_data[filtered_data['country'].isin(selected_countries)]
    if score_range:
        filtered_data = filtered_data[(filtered_data['Average_Score'] >= score_range[0]) &
                                      (filtered_data['Average_Score'] <= score_range[1])]
    return filtered_data.to_dict('records')

@app.callback(
    dash.dependencies.Output('image', 'src'),
    [dash.dependencies.Input('image-dropdown', 'value')])
def update_image_src(value):
    return static_image_route + value

@app.server.route('{}<image_path>.png'.format(static_image_route))
def serve_image(image_path):
    image_name = '{}.png'.format(image_path)
    if image_name not in list_of_images:
        raise Exception('"{}" is excluded from the allowed static files'.format(image_path))
    return flask.send_from_directory(image_directory, image_name)

if __name__ == '__main__':
    app.run_server(debug=True)