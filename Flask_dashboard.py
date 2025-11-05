import sqlite3
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from flask import Flask

server = Flask(__name__)
app = dash.Dash(__name__, server=server, routes_pathname_prefix='/dashboard/')

def fetch_mood_data():
    conn = sqlite3.connect("mood_tracking.db")
    df = pd.read_sql_query("SELECT * FROM mood_log ORDER BY timestamp DESC LIMIT 100", conn)
    conn.close()
    return df

def process_mood_trends():
    df = fetch_mood_data()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['day'] = df['timestamp'].dt.date
    mood_counts = df.groupby(['day', 'emotion']).size().reset_index(name='count')
    return mood_counts

app.layout = html.Div([
    html.H1("Employee Mood Dashboard", style={'textAlign': 'center'}),
    
    dcc.Interval(
        id='interval-component',
        interval=60000,  
        
        n_intervals=0
    ),
    
    html.Div([
        dcc.Graph(id='mood-trend-line'),
        dcc.Graph(id='mood-distribution-pie')
    ])
])

@app.callback(
    [Output('mood-trend-line', 'figure'), Output('mood-distribution-pie', 'figure')],
    Input('interval-component', 'n_intervals')
)
def update_graphs(n):
    mood_trends = process_mood_trends()
    
    line_fig = px.line(mood_trends, x='day', y='count', color='emotion', title='Mood Trends Over Time')
    pie_fig = px.pie(mood_trends, names='emotion', values='count', title='Mood Distribution (Past Week)')
    
    return line_fig, pie_fig

if __name__ == '__main__':
    app.run(debug=True)
