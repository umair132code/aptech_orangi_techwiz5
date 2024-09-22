import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import calendar


data = pd.read_csv('GlobalTemperatures.csv')
data['dt'] = pd.to_datetime(data['dt'])
data = data.dropna()


data['LandAverageTemperatureUncertainty'] = data.get('LandAverageTemperatureUncertainty', 0)
data['LandMaxTemperatureUncertainty'] = data.get('LandMaxTemperatureUncertainty', 0)
data['LandMinTemperatureUncertainty'] = data.get('LandMinTemperatureUncertainty', 0)
data['LandAndOceanAverageTemperatureUncertainty'] = data.get('LandAndOceanAverageTemperatureUncertainty', 0)


data['Year'] = data['dt'].dt.year
data['Month'] = data['dt'].dt.month


X = data[['Year', 'Month', 'LandMaxTemperature', 'LandMinTemperature', 'LandAndOceanAverageTemperature']]
y = data['LandAverageTemperature']

# train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


app = Dash(__name__)


def create_historical_land_ocean_temp_chart():
    return dcc.Graph(
        id='historical-land-ocean-temp',
        figure=go.Figure([
            go.Scatter(x=data['dt'], y=data['LandAndOceanAverageTemperature'], mode='lines', name='Land & Ocean Avg Temp', line=dict(color='green')),
            go.Scatter(
                x=data['dt'],
                y=data['LandAndOceanAverageTemperature'] + data['LandAndOceanAverageTemperatureUncertainty'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(0,255,0,0.1)',
                name='Uncertainty Range'
            ),
            go.Scatter(
                x=data['dt'],
                y=data['LandAndOceanAverageTemperature'] - data['LandAndOceanAverageTemperatureUncertainty'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(0,255,0,0.1)'
            )
        ]).update_layout(title='Historical Land and Ocean Average Temperatures Over Time')
    )


def create_static_charts():
    # Chart 1: Global Land Average Temperature Over Time
    chart_1 = dcc.Graph(
        id='global-land-avg-temp',
        figure=go.Figure([
            go.Scatter(x=data['dt'], y=data['LandAverageTemperature'], mode='lines', name='Land Avg Temp'),
            go.Scatter(
                x=data['dt'],
                y=data['LandAverageTemperature'] + data['LandAverageTemperatureUncertainty'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(0,100,255,0.1)',
                name='Uncertainty Range'
            ),
            go.Scatter(
                x=data['dt'],
                y=data['LandAverageTemperature'] - data['LandAverageTemperatureUncertainty'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(0,100,255,0.1)'
            )
        ]).update_layout(title='Global Land Average Temperature Over Time')
    )

    
    chart_2 = dcc.Graph(
        id='global-max-vs-min-temp',
        figure=go.Figure([
            # Max Temperature with Uncertainty
            go.Scatter(x=data['dt'], y=data['LandMaxTemperature'], mode='lines', name='Max Temperature', line=dict(color='red')),
            go.Scatter(
                x=data['dt'],
                y=data['LandMaxTemperature'] + data['LandMaxTemperatureUncertainty'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.1)',
                name='Max Temperature Uncertainty'
            ),
            go.Scatter(
                x=data['dt'],
                y=data['LandMaxTemperature'] - data['LandMaxTemperatureUncertainty'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.1)'
            ),
            # Min Temperature with Uncertainty
            go.Scatter(x=data['dt'], y=data['LandMinTemperature'], mode='lines', name='Min Temperature', line=dict(color='blue', dash='dash')),
            go.Scatter(
                x=data['dt'],
                y=data['LandMinTemperature'] + data['LandMinTemperatureUncertainty'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(0,0,255,0.1)',
                name='Min Temperature Uncertainty'
            ),
            go.Scatter(
                x=data['dt'],
                y=data['LandMinTemperature'] - data['LandMinTemperatureUncertainty'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(0,0,255,0.1)'
            )
        ]).update_layout(title='Global Max vs Min Land Temperatures Over Time with Uncertainty')
    )

    y_pred = model.predict(X_test)
    chart_3 = dcc.Graph(
        id='predicted-vs-actual-temp',
        figure=go.Figure([
            go.Scatter(x=X_test['Year'], y=y_test, mode='markers', name='Actual Temperatures', marker=dict(color='blue')),
            go.Scatter(x=X_test['Year'], y=y_pred, mode='markers', name='Predicted Temperatures', marker=dict(color='red', symbol='circle-open'))
        ]).update_layout(title='Actual vs Predicted Temperatures')
    )

    chart_4 = create_historical_land_ocean_temp_chart()

    return html.Div([chart_1, chart_2, chart_3, chart_4])


app.layout = html.Div([
    html.H1("Planet Heat"),
    create_static_charts(),
    html.Br(),
    html.Label("Enter the year you want to predict temperatures for:"),
    dcc.Input(id='year-input', type='number', value=2050, min=1750, max=2100, step=1),
    html.Br(),
    html.Br(),
    html.Div(id='dynamic-charts-output')
])

@app.callback(
    Output('dynamic-charts-output', 'children'),
    [Input('year-input', 'value')]
)
def update_dynamic_charts(user_year):
    # Predict temperatures for the user-specified year
    future_year = pd.DataFrame({
        'Year': [user_year] * 12,  # Predict for all months of the specified year
        'Month': list(range(1, 13)),  # Months from January to December
        'LandMaxTemperature': [25] * 12,  # Example values, adjust based on past trends
        'LandMinTemperature': [15] * 12,  # Example values
        'LandAndOceanAverageTemperature': [20] * 12  # Example value, adjust as needed
    })

 
    future_year.fillna(0, inplace=True)

   
    future_temp_predictions = model.predict(future_year)

  
    future_year['MonthName'] = future_year['Month'].apply(lambda x: calendar.month_name[x])

    
    future_year['PredictedTempC'] = future_temp_predictions
    future_year['PredictedTempF'] = future_temp_predictions * 9/5 + 32



    # Chart 1: Predicted Temperatures (18°C to 22°C)
    filtered_future_year = future_year[(future_year['PredictedTempC'] >= 18) & (future_year['PredictedTempC'] <= 22)]
    chart_1 = dcc.Graph(
        id='predicted-temp',
        figure=go.Figure([
            go.Scatter(x=filtered_future_year['MonthName'], y=filtered_future_year['PredictedTempC'], mode='markers', name='Predicted Temp (°C)')
        ]).update_layout(title=f'Predicted Temperatures for {user_year} (18°C to 22°C)')
    )

    chart_2 = dcc.Graph(
        id='all-predicted-temp',
        figure=go.Figure([
            go.Scatter(x=future_year['MonthName'], y=future_year['PredictedTempC'], mode='lines+markers', name='Predicted Temp (°C)'),
            go.Scatter(x=future_year['MonthName'], y=future_year['LandAndOceanAverageTemperature'], mode='lines', name='Predicted Land & Ocean Avg Temp', line=dict(color='green'))
        ]).update_layout(title=f'All Predicted Temperatures for {user_year}')
    )

    
    return html.Div([chart_1, chart_2])

if __name__ == '__main__':
    app.run_server(debug=True)
