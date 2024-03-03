import numpy as np
import pandas as pd
import datetime as dt

import plotly.express as px
import plotly.graph_objects as go

import yfinance as yf

from pflib import *

from io import StringIO

from dash import html, dcc, Input, Output, Dash
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from dash.exceptions import PreventUpdate

    
def evaluate_portfolio(mc_portfolios, index, data, initialValue):
    portfolio = mc_portfolios.loc[index]
    tickers = data.columns
    nShares = portfolio[[ticker+' weight' for ticker in tickers]].rename({ticker+' weight' : ticker for ticker in tickers})*initialValue/data.iloc[0]
    portfolio_value = nShares.dot(data.T)
    return portfolio_value

def evaluate_asset(tickers, index, data, initialValue):
    asset = data.iloc[:, index] if len(tickers) > 1 else data
    nShares = initialValue/asset.iloc[0]
    asset_value = nShares*asset
    return asset_value

# Get a list of symbols from FTSEMIB index
ftsemib = pd.read_html('https://en.wikipedia.org/wiki/FTSE_MIB')[1]
ftsemib['ICB Sector'] = ftsemib['ICB Sector'].str.extract(r'\((.*?)\)', expand=False).fillna(ftsemib['ICB Sector'])


dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
load_figure_template("CERULEAN")

app = Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN, dbc_css])

app.layout = html.Div([
    dcc.Store(id='store-data'),
    dcc.Store(id='store-portfolios'),
    html.Div(className='dbc',
             children=[
                 dbc.Row(
                     children=[
                         dbc.Col(
                             width=2,
                             children=[dbc.Card([
                                 html.P('Select assets', className='dbc'),
                                 dcc.Dropdown(
                                     id='ticker-dropdown',
                                     options=[
                                         {'label': f"{row['Company']} ({row['Ticker']})", 'value': row['Ticker']}
                                         for _, row in ftsemib.iterrows()
                                     ],
                                     multi=True,
                                     className='dbc'
                                 ),
                                 html.Br(),

                                 html.P('Select start date',
                                        className='dbc'),
                                 dcc.DatePickerSingle(
                                     id='start-date',
                                     min_date_allowed=dt.date(2010, 1, 1),
                                     max_date_allowed=dt.date.today() - dt.timedelta(days=365),
                                     initial_visible_month=dt.date.today() - dt.timedelta(days=365),
                                     date=dt.date.today() - dt.timedelta(days=365),
                                     display_format='DD/MM/YYYY',
                                     className='dbc'
                                 ),

                                 html.Br(),
                                 html.P('Analysis window',
                                        className='dbc'),
                                 dcc.Slider(
                                     id='analysis-window',
                                     min=1,
                                     max=10,
                                     step=1,
                                     value=1,
                                     marks=None,
                                     tooltip={
                                         'placement': 'bottom',
                                         'template': '{value} years',
                                     },
                                     className='dbc'
                                 ),
                                html.Br(),
                                html.P('Risk-free rate', className='dbc'),
                                dbc.Input(
                                    id='risk-free-rate',
                                    value=4,
                                    type='number',
                                    step=0.1,
                                    className='dbc',
                                 ),
                                 html.Br(),
                                html.P('Options', className='dbc'),
                                dbc.Switch(
                                    id='include-risk-free',
                                    label='Include risk-free investment',
                                    value=False,
                                    className='dbc'
                                ),
                                dbc.Switch(
                                    id='short-selling',
                                    label='Allow short selling',
                                    value=False,
                                    className='dbc'
                                ),
                                 html.Br(),
                                 html.P('Number of samples', className='dbc'),
                                 dbc.Input(id='n-portfolios', value=1000, type='number', className='dbc'),

                                 dbc.Button('Run',
                                            id='run-button',
                                            n_clicks=0,
                                            className='dbc'),

                                 html.Br(),

                                 html.P('Initial investment', className='dbc'),
                                 dbc.Input(id='initial-investment', value=100, type='number', className='dbc'),
                                 dbc.Switch(id='show-short', label='Show short positions', value=False, className='dbc'),
                             ])]
                         ),
                         dbc.Col(
                             width=10,
                             children=[
                                 dcc.Tabs(
                                     id='tabs',
                                     value='tab-1',
                                     className='dbc',
                                     children=[
                                         dcc.Tab(
                                             label='Portfolio selection',
                                             id='tab-1',
                                             className='dbc',
                                             children=[
                                                 dbc.Row([

                                                         dbc.Col(
                                                            children = dcc.Graph(id='markowitz-graph',
                                                                                figure=go.Figure(),
                                                                       clear_on_unhover=True,
                                                                       )
                                                         ),

                                                         dbc.Col(
                                                             children=[
                                                                 dcc.Graph(id='portfolio-value',
                                                                           figure=go.Figure(),
                                                                           
                                                                   )
                                                             ]
                                                         )
                                                 ]),

                                             ]
                                         ),
                                         dcc.Tab(
                                             id='tab-3',
                                             label='Realized returns',
                                             children=[
                                                 html.Div('Work in progress', className='dbc')
                                             ]
                                         ),
                                     ]
                                 ),
                             ]
                         ),
                     ],
                 )
             ]
             ),


])


# Download data and plot mean-variance graph for selected assets
@app.callback(
    [Output('markowitz-graph', 'figure'),
     Output('store-data', 'data')],
    [Input('ticker-dropdown', 'value'),
     Input('start-date', 'date'),
     Input('analysis-window', 'value'),
     Input('risk-free-rate', 'value'),],
    # prevent_initial_call=True,
    # suppress_callback_exceptions=True
)
def select_assets(tickers, investment_start_date, window, riskFreeRate):
    riskFreeRate = riskFreeRate/100
    if not tickers:
        fig = go.Figure().update_xaxes(title='Risk', range=[0, 0.5]).update_yaxes(title='Return',range=[0, 0.4]).update_layout(transition_duration=500)
        data = pd.DataFrame()
        return fig, data.to_json()

    investment_start_date = dt.datetime.strptime(investment_start_date, '%Y-%m-%d')
    # Analyse assets over a window prior to the start date
    start_date = investment_start_date - dt.timedelta(days=window * 365)
    # Evaluate the investment over one year after the start date
    end_date = investment_start_date + dt.timedelta(days=365)

    basket = Basket(tickers, riskFreeRate)
    basket.get_data(start_date, end_date)
    data = basket.data

    stocks_mv, minVar_portfolio, maxSharpe_portfolio, efficient_frontier = basket.mv_analysis(investment_start_date)

    fig = px.scatter(stocks_mv, x='Risk', y='Return', text=stocks_mv.index).update_traces(marker=dict(size=10), name='Stocks')

    if len(tickers) > 1:
        # Add minimum variance portfolio
        fig.add_scatter(x=[minVar_portfolio['risk']], y=[minVar_portfolio['return']], mode='markers',
                                 marker=dict(size=10), showlegend=False,
                                 name='Minimum variance portfolio', text='Minimum variance portfolio')
        # Add max Sharpe portfolio
        fig.add_scatter(x=[maxSharpe_portfolio['risk']], y=[maxSharpe_portfolio['return']], mode='markers',
                                 marker=dict(size=10, color='red'), showlegend=False,
                                 name='Max Sharpe portfolio', text='Max Sharpe portfolio')
        # Add efficient frontier
        for i in range(len(fig['data'])):
            if fig['data'][i]['name'] == 'Minimum variance portfolio':
                color = fig['data'][i]['marker']['color']
        fig.add_scatter(x=efficient_frontier['Risk'], y=efficient_frontier['Return'], mode='lines', 
                     line=dict(color=color, width=1), 
                     name='Minimum variance line', showlegend=False)
        # Add risk-free asset
        for i in range(len(fig['data'])):
            if fig['data'][i]['name'] == 'Max Sharpe portfolio':
                color = fig['data'][i]['marker']['color']
        fig.add_scatter(x=[0], y=[riskFreeRate], mode='markers',
                                 marker=dict(size=10, color=color), showlegend=False,
                                 name='Risk-free asset', text='Risk-free asset')
        # Add market line
        fig.add_scatter(x=[0, maxSharpe_portfolio['risk']], y=[riskFreeRate, maxSharpe_portfolio['return']],
                                 mode='lines', line=dict(color=color, width=1), showlegend=False,
                                 name='Capital market line', text='Capital market line')
        
        fig.update_xaxes(range=[0, 0.5])
    # Market line and efficient frontier based on ylims
    fig.update_traces(textposition='top center').update_layout(transition_duration=100, title='Asset selection')

    return fig, data.to_json()


@app.callback(
    [Output('markowitz-graph', 'figure', allow_duplicate=True),
     Output('store-portfolios', 'data'),
     Output('ticker-dropdown', 'disabled'),
     Output('run-button', 'n_clicks')],
    [Input('ticker-dropdown', 'value'),
     Input('risk-free-rate', 'value'),
     Input('n-portfolios', 'value'),
     Input('start-date', 'date'),
     Input('analysis-window', 'value'),
     Input('include-risk-free', 'value'),
     Input('short-selling', 'value'),
     Input('show-short', 'value'),
     Input('run-button', 'n_clicks'),
     ],
    prevent_initial_call=True,
    suppress_callback_exceptions=True
)
def mc_allocation(tickers, riskFreeRate, n_portfolios, investment_start_date, window, includeRiskFree, shortSelling, showShort, n_clicks):
    riskFreeRate = riskFreeRate/100
    if not tickers:
        fig = go.Figure()
        return fig, None, False, None
    
    if not n_clicks:
        raise PreventUpdate

    
    investment_start_date = dt.datetime.strptime(investment_start_date, '%Y-%m-%d')
    # Analyse assets over a window prior to the start date
    start_date = investment_start_date - dt.timedelta(days=window * 365)
    # Evaluate the investment over one year after the start date
    end_date = investment_start_date + dt.timedelta(days=365)

    basket = Basket(tickers, riskFreeRate)
    basket.get_data(start_date, end_date)
    data = basket.data

    # Check if only one asset is selected by checking if data is a Series
    if isinstance(data, pd.Series):
        # TODO - Warn user that only one asset is selected
        raise PreventUpdate
    analysis_returns = data[:investment_start_date].pct_change().dropna()

    if analysis_returns.empty:
        fig = go.Figure().update_layout(transition_duration=500)
        n_clicks = None
        mc_portfolios = pd.DataFrame()
        assetInputDisabled = True
        return fig, mc_portfolios.to_json(), assetInputDisabled, n_clicks

    stocks_mv, minVar_portfolio, maxSharpe_portfolio, efficient_frontier = basket.mv_analysis(investment_start_date)

    portfolio = Portfolio(basket, riskFreeRate, includeRiskFree)
    portfolio.set_investment_start(investment_start_date)
    mc_portfolios = portfolio.generate(n_portfolios, shortSelling)

    assetList = tickers + ['Risk-free'] if includeRiskFree else tickers

    # Plot the random portfolios
    color = 'Short positions' if showShort else 'Sharpe Ratio'
    fig = px.scatter(mc_portfolios, x='Risk', y='Return', color=color, hover_data={**{asset +' weight': ':.2f' for asset in assetList}, **{'Return': ':.2f', 'Risk': ':.2f', 'Sharpe Ratio': ':.2f'}}, opacity=0.5,).update_traces(name='Monte Carlo samples')
    
    # Add the stocks
    fig.add_scatter(x=stocks_mv['Risk'], y=stocks_mv['Return'], mode='markers', marker=dict(size=7.5,),showlegend=False, name='Stocks', text = [f'<b>{index}</b> <br>Standard deviation: {vol:.2f}<br>Expected return: {ret:.2f}' for index, vol, ret in zip(stocks_mv.index, stocks_mv['Risk'], stocks_mv['Return'])],hoverinfo='text')
    
    # Add minimum variance portfolio
    fig.add_scatter(x=[minVar_portfolio['risk']], y=[minVar_portfolio['return']], mode='markers',
                            marker=dict(size=10,), showlegend=False,
                            name='Minimum variance portfolio', text='Minimum variance portfolio')
    # Add max Sharpe portfolio
    fig.add_scatter(x=[maxSharpe_portfolio['risk']], y=[maxSharpe_portfolio['return']], mode='markers',
                                 marker=dict(size=10, color='red'), showlegend=False,
                                 name='Market portfolio', text='Market portfolio')
    # Add efficient frontier
    for i in range(len(fig['data'])):
        if fig['data'][i]['name'] == 'Minimum variance portfolio':
            color = fig['data'][i]['marker']['color']
    fig.add_scatter(x=efficient_frontier['Risk'], y=efficient_frontier['Return'], mode='lines', 
                                 line=dict(color=color,width=1), name='Minimum variance line', showlegend=False)

    if includeRiskFree:
        for i in range(len(fig['data'])):
            if fig['data'][i]['name'] == 'Max Sharpe portfolio':
                color = fig['data'][i]['marker']['color']
        # Add risk-free asset
        fig.add_scatter(x=[0], y=[riskFreeRate], mode='markers',
                                marker=dict(size=10, color=color), showlegend=False,
                                name='Risk-free asset', text='Risk-free asset')
        # Add market line
        fig.add_scatter(x=[0, maxSharpe_portfolio['risk']], y=[riskFreeRate, maxSharpe_portfolio['return']],
                                 mode='lines', line=dict(color=color, width=1), showlegend=False,
                                 name='Capital market line', text='Capital market line')

    if len(mc_portfolios) <= 1000:
        fig.update_layout(transition_duration=500)

    # Set axis limits
    x_min = 0 if includeRiskFree else minVar_portfolio['risk']
    x_max = max(stocks_mv['Risk'].max(), maxSharpe_portfolio['risk'])
    xlims = [0.9*x_min, x_max*1.1]
    # Set y-axis limits
    y_min = min(stocks_mv['Return'].min(), 0) if includeRiskFree else stocks_mv['Return'].min()
    coeff = 0.9 if y_min > 0 else 1.1
    y_max = max(maxSharpe_portfolio['return'], stocks_mv['Return'].max())
    ylims = [coeff*y_min, y_max*1.1]

    fig.update_xaxes(range=xlims).update_yaxes(range=ylims)

    fig.update_layout(title='Monte Carlo Simulation')

    n_clicks = None

    assetInputDisabled = True

    return fig, mc_portfolios.to_json(), assetInputDisabled, n_clicks


@app.callback(
    Output('portfolio-value', 'figure'),
    [Input('ticker-dropdown', 'value'),
    Input('risk-free-rate', 'value'),
    Input('analysis-window', 'value'),
    Input('include-risk-free', 'value'),
    Input('short-selling', 'value'),
    Input('store-portfolios', 'data'),
    Input('markowitz-graph', 'clickData'),
    Input('markowitz-graph', 'hoverData'),
    Input('markowitz-graph', 'figure'),
    Input('initial-investment', 'value'),
    Input('start-date', 'date'),],
)
def plot_portfolio(tickers, riskFreeRate, window, includeRiskFree, shortSelling, mcPortfolios, clickData, hoverData, figure, initial_investment, investment_start_date):
    if not tickers:
        raise PreventUpdate
    riskFreeRate = riskFreeRate/100
    
    investment_start_date = dt.datetime.strptime(investment_start_date, '%Y-%m-%d')
    # Analyse assets over a window prior to the start date
    start_date = investment_start_date - dt.timedelta(days=window * 365)
    # Evaluate the investment over one year after the start date
    end_date = investment_start_date + dt.timedelta(days=365)

    basket = Basket(tickers, riskFreeRate)
    basket.get_data(start_date, end_date)
    portfolio = Portfolio(basket, riskFreeRate, includeRiskFree)

    outOfSampleData = basket.data[investment_start_date:]
    ylims = [((initial_investment/outOfSampleData.iloc[0])*outOfSampleData.min()).min(), ((initial_investment/outOfSampleData.iloc[0])*outOfSampleData.max()).max()]
    fig = go.Figure()
    
    if not clickData and not hoverData:
        raise PreventUpdate

    if not mcPortfolios:
        if clickData:
            curveNumber = clickData['points'][0]['curveNumber']
            trace_name = figure['data'][curveNumber]['name']
            if trace_name != 'Stocks':
                raise PreventUpdate
            index = clickData['points'][0]['pointNumber']
            asset_value = portfolio.basket.stocks[index].evaluate(initial_investment, investment_start_date)
            fig.add_trace(go.Scatter(x=asset_value.index, y=asset_value, mode='lines', line=dict(color='black')))

        if hoverData:
            curveNumber = hoverData['points'][0]['curveNumber']
            trace_name = figure['data'][curveNumber]['name']
            if trace_name != 'Stocks':
                raise PreventUpdate
            index = hoverData['points'][0]['pointNumber']
            asset_value = portfolio.basket.stocks[index].evaluate(initial_investment, investment_start_date)
            fig.add_trace(go.Scatter(x=asset_value.index, y=asset_value, mode='lines', opacity=0.3, line=dict(color='black')))

        fig.update_yaxes(range=ylims).update_layout(showlegend=False, title='Portfolio value')
        # else:
        #     raise PreventUpdate
    else:
        portfolio.load(mcPortfolios)
        portfolio.set_investment_start(investment_start_date)

        if clickData:
            index = clickData['points'][0]['pointNumber']
            curveNumber = clickData['points'][0]['curveNumber']
            trace_name = figure['data'][curveNumber]['name']
            if trace_name == 'Monte Carlo samples':
                portfolio_value = portfolio.evaluate(index, initial_investment)
                fig = px.line(portfolio_value)
            elif trace_name == 'Stocks':
                asset_value = portfolio.basket.stocks[index].evaluate(initial_investment, investment_start_date)
                fig = px.line(asset_value).update_traces(line_color='black')

        
        if hoverData:
            index = hoverData['points'][0]['pointNumber']
            curveNumber = hoverData['points'][0]['curveNumber']
            trace_name = figure['data'][curveNumber]['name']
            if trace_name == 'Monte Carlo samples':
                portfolio_value = portfolio.evaluate(index, initial_investment)
                fig.add_trace(go.Scatter(x=portfolio_value.index, y=portfolio_value, mode='lines', opacity=0.3))
            elif trace_name == 'Assets':
                asset_value = portfolio.basket.stocks[index].evaluate(initial_investment, investment_start_date)
                fig.add_trace(go.Scatter(x=asset_value.index, y=asset_value, mode='lines', opacity=0.3, line=dict(color='black')))

        fig.update_layout(showlegend=False, title='Portfolio value')
        fig.update_yaxes(range=ylims) if not shortSelling else None
    
    return fig

# import json

# @app.callback(
#     Output('hover-data', 'children'),
#     Input('markowitz-graph', 'hoverData'))
# def hover_data(hoverData):
#     return json.dumps(hoverData, indent=2)

# Delete before deploying
if __name__ == '__main__':
    app.run_server(debug=True,)