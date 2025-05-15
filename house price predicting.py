# Full updated code for the Dash app that:
# - Loads data from a local path
# - Calculates average price per sqft
# - Allows user to input only Square Feet and calculates estimated price
# - Includes model-based predictions and visualizations

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import os

# Load dataset
df = pd.read_csv("D:\data\data.csv").dropna()
df['price_per_sqft'] = df['price'] / df['square_feet']
bins = [df['price'].min(), df['price'].quantile(0.33), df['price'].quantile(0.66), df['price'].max()]
df['price_band'] = pd.cut(df['price'], bins=bins, labels=['Low', 'Medium', 'High'])

# Features & target
features = ['square_feet', 'num_rooms', 'location_score']
target = 'price'

X = df[features].values
y = df[target].values

scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

avg_price_per_sqft = df['price_per_sqft'].mean()

# Initialize app
app = Dash(__name__)
app.title = "🏡 Virtual Home - Price Explorer"

app.layout = html.Div([
    html.H1("🏡 Virtual Home: House Price Explorer", style={'textAlign': 'center'}),

    html.Div([
        html.H3("🔧 Settings"),
        html.Label("Choose a model:"),
        dcc.RadioItems(
            id='model-choice',
            options=[
                {'label': 'Linear Regression', 'value': 'lr'},
                {'label': 'Random Forest', 'value': 'rf'}
            ],
            value='lr',
            inline=True
        ),
        html.Label("Color houses by:"),
        dcc.Dropdown(
            id='color-by',
            options=[
                {'label': 'Actual Price', 'value': 'price'},
                {'label': 'Price per SqFt', 'value': 'price_per_sqft'},
                {'label': 'Price Band', 'value': 'price_band'}
            ],
            value='price',
            clearable=False
        ),
        html.Br(),
        html.H3("📝 Your House Details"),
        html.Label("Square Feet:"), dcc.Input(id='input-sqft', type='number', placeholder='e.g. 1200'),
        html.Br(), html.Label("Number of Rooms:"), dcc.Input(id='input-rooms', type='number', placeholder='e.g. 3'),
        html.Br(), html.Label("Location Score:"), dcc.Input(id='input-location', type='number', placeholder='e.g. 7'),
        html.Br(), html.Button('Predict Price', id='predict-button', n_clicks=0),
        html.Div(id='prediction-output', style={'marginTop': '20px', 'fontSize': '18px'}),
        html.Br(),

        html.H3("🧮 Sq.Ft. Price Calculator"),
        html.Label("Enter Square Feet:"), dcc.Input(id='calc-sqft', type='number', step=1),
        html.Br(), html.Button('Calculate Total Price', id='calc-button', n_clicks=0),
        html.Div(id='calc-result', style={'marginTop': '15px', 'fontSize': '18px'}),
        html.Br(),

        html.H4("🗣️ Share your opinion:"),
        dcc.Textarea(id='user-opinion', placeholder='Type your opinion (e.g. I value number of rooms most)', style={'width': '100%', 'height': 100}),
        html.Button('Submit Opinion', id='submit-opinion', n_clicks=0),
        html.P("Click points in the scatter to view details.")
    ], style={'width': '30%', 'display': 'inline-block', 'padding': '20px'}),

    html.Div([
        dcc.Graph(id='3d-scatter'),
        dcc.Graph(id='bar-feature-importance'),
        dcc.Graph(id='heatmap-corr'),
        dcc.Graph(id='opinion-based-graph')
    ], style={'width': '68%', 'display': 'inline-block', 'padding': '10px'}),

    html.Div(id='house-details', style={'padding': '20px', 'fontSize': '18px'}),
    html.Div(id='model-metrics', style={'padding': '20px', 'backgroundColor': '#f9f9f9', 'fontSize': '16px'})
])

@app.callback(
    Output('3d-scatter', 'figure'),
    Output('bar-feature-importance', 'figure'),
    Output('heatmap-corr', 'figure'),
    Output('opinion-based-graph', 'figure'),
    Output('house-details', 'children'),
    Output('model-metrics', 'children'),
    Input('model-choice', 'value'),
    Input('color-by', 'value'),
    Input('3d-scatter', 'clickData'),
    Input('submit-opinion', 'n_clicks'),
    State('user-opinion', 'value')
)
def update_graphs(model_choice, color_by, clickData, n_clicks_opinion, opinion_text):
    model = LinearRegression() if model_choice == 'lr' else RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    scatter_fig = px.scatter_3d(
        df, x='square_feet', y='num_rooms', z='location_score',
        color=df[color_by] if color_by != 'price_band' else df['price_band'].astype(str),
        title="3D Visualization of Homes",
        size_max=12,
        labels={'square_feet': 'Sq Ft', 'num_rooms': 'Rooms', 'location_score': 'Location Score'}
    )

    importances = model.coef_ if model_choice == 'lr' else model.feature_importances_
    bar_fig = go.Figure([
        go.Bar(x=features, y=importances, marker_color='lightskyblue')
    ])
    bar_fig.update_layout(title="📊 Feature Impact on Price", yaxis_title="Importance")

    corr = df[features + [target]].corr()
    heatmap_fig = px.imshow(corr, text_auto=True, title="📈 Feature Correlation Matrix")

    if opinion_text:
        keyword = 'num_rooms' if 'room' in opinion_text.lower() else 'square_feet' if 'size' in opinion_text.lower() else 'location_score'
    else:
        keyword = 'square_feet'
    opinion_fig = px.histogram(df, x=keyword, title=f"📌 Based on Your Opinion: Distribution of {keyword.capitalize()}")

    if clickData:
        pt = clickData['points'][0]
        details = html.Div([
            html.H4("🏠 House Details:"),
            html.P(f"• Square Feet: {pt['x']}"),
            html.P(f"• Rooms: {pt['y']}"),
            html.P(f"• Location Score: {pt['z']}"),
            html.P(f"• Estimated Price: ₹{pt['marker.color']:,.2f}" if isinstance(pt['marker']['color'], (int, float)) else "")
        ])
    else:
        details = "Click a house in the graph to view details."

    metrics = html.Div([
        html.H4("📊 Model Metrics"),
        html.P(f"• Mean Absolute Error (MAE): ₹{mae:,.2f}"),
        html.P(f"• R² Score: {r2:.3f}")
    ])

    return scatter_fig, bar_fig, heatmap_fig, opinion_fig, details, metrics

@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('input-sqft', 'value'),
    State('input-rooms', 'value'),
    State('input-location', 'value'),
    State('model-choice', 'value')
)
def predict_price(n_clicks, sqft, rooms, location_score, model_choice):
    if n_clicks > 0 and sqft and rooms and location_score:
        user_input = np.array([[sqft, rooms, location_score]])
        scaled_input = scaler.transform(user_input)
        model = LinearRegression() if model_choice == 'lr' else RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        predicted = model.predict(scaled_input)[0]
        return f"💰 Predicted Price: ₹{predicted:,.2f}"
    return ""

@app.callback(
    Output('calc-result', 'children'),
    Input('calc-button', 'n_clicks'),
    State('calc-sqft', 'value')
)
def calc_price(n_clicks, sqft):
    if n_clicks > 0 and sqft:
        total = sqft * avg_price_per_sqft
        return html.Div([
            html.P(f"📏 Average Rate: ₹{avg_price_per_sqft:,.2f} per Sq.Ft."),
            html.P(f"🏷️ Estimated Total Price: ₹{total:,.2f}")
        ])
    return ""

if __name__ == '__main__':
    app.run(debug=True)

