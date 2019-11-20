import copy
import plotly.graph_objs as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from ecg.utils import heart_rate
from utils import parse_data
from plotting import plot_raw, plot_hist, plot_pointcarre, plot_frequency

# filename = 'C:/Users/au646069/Desktop/dash/Subject_1111398.npy'

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H1(
            "Heart Rate Variability Toolbox",
            style={"margin-bottom": "0px", 'textAlign': 'center'})],
        id="header",
        className="row flex-display",
        style={"margin-bottom": "25px"},
    ),

    ############
    # Upper menu
    ############
    html.Div(
        [
            # Drag and drop files
            html.Div([

                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Files')
                    ]),
                    # Allow multiple files to be uploaded
                    multiple=True
                ),
            ],
                id='dragdrop-container',
                className='pretty_container'
            ),

            # Dropdown menu - file selection
            html.Div([
                dcc.Dropdown(
                    id='dropdown-files',
                    options=[
                        {'label': 'No files availables', 'value': ''},
                    ],
                    value=[],
                    multi=True,
                    placeholder="Select files",
                ),
            ],
                id='dropdownfiles-container',
                className='pretty_container three columns')
        ],
        className="row flex-display",
    ),

    ###############
    # First section
    ###############
    html.Div(
        [

            ###########
            # Histogram
            ###########
            html.Div(
                [

                    html.P("Heart rate metric:",
                           className="control_label"),
                    dcc.RadioItems(
                        id="hr_metric",
                        options=[
                            {"label": "R-R ", "value": "rr"},
                            {"label": "BPM", "value": "bpm"},
                        ],
                        value="active",
                        labelStyle={"display": "inline-block"},
                        className="dcc_control",
                    ),

                    dcc.Graph(id="histogram")

                ],
                id="histogramContainer",
                className="pretty_container four columns",
            ),

            ######################
            # Main raw data viewer
            ######################
            html.Div(
                [

                    dcc.Graph(id="Raw"),

                ],
                id="RawContainer",
                className="pretty_container eigth columns",
            ),

        ],
        className="row flex-display",
    ),

    ################
    # Second section
    ###############
    html.Div(
        [
            #############
            # Time domain
            #############
            html.Div(
                [dcc.Graph(id="timeDomain")],
                id="timeDomainContainer",
                className="pretty_container eight columns",
            ),

            ##################
            # Frequency domain
            ##################
            html.Div(
                [dcc.Graph(id="frequencyDomain")],
                id="frequencyDomainContainer",
                className="pretty_container four columns",
            ),

        ],
        className="row flex-display",
    ),

    ###############
    # Third section
    ###############
    html.Div(
        [
            ############
            # Pointcarre
            ############
            html.Div(
                [dcc.Graph(id="pointcarrePlot")],
                id="pointcarreContainer",
                className="pretty_container four columns",
            ),

            ##################
            # Frequency domain
            ##################
            html.Div(
                [dcc.Graph(id="nonlinear")],
                id="nonlinearContainer",
                className="pretty_container eight columns",
            ),

        ],
        className="row flex-display",
    ),


    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)


# Data uploading
@app.callback(Output('dropdown-files', 'options'),
              [Input('upload-data', 'filename')])
def update_data(filenames):
    options = [{'label': 'No files availables', 'value': ''}]
    if filenames:
        options = []
        for f in filenames:
            options.append({'label': f, 'value': f})

    return options


# Dropdown menu - File selection
@app.callback(
    [Output('Raw', 'figure'),
     Output('histogram', 'figure'),
     Output('frequencyDomain', 'figure'),
     Output('pointcarrePlot', 'figure')],
    [dash.dependencies.Input('upload-data', 'filename'),
     dash.dependencies.Input('dropdown-files', 'value'),
     dash.dependencies.Input('hr_metric', 'value')])
def update_output(filenames, selected, hr_metric):

    df = None
    # Read data
    if filenames:
        if selected:
            df = parse_data(selected[0], hr_metric=hr_metric)
        else:
            df = parse_data(filenames[0], hr_metric=hr_metric)

    if df is not None:

        # Update main plot
        raw = plot_raw(df)

        # Update histogram
        hist = plot_hist(df)

        # Update pointcarre plot
        pointcarrePlot = plot_pointcarre(df)

        # Update frequency
        frequencyPlot = plot_frequency(df)

    else:
        # Create empty plots
        raw = dict(data=[], layout=dict())
        hist = dict(data=[], layout=dict())
        frequencyPlot = dict(data=[], layout=dict())
        frequencyPlot = dict(data=[], layout=dict())
        pointcarrePlot = dict(data=[], layout=dict())

    return raw, hist, frequencyPlot, pointcarrePlot


if __name__ == '__main__':
    app.run_server(debug=True)
