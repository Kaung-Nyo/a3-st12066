# Import required libraries
import numpy as np
import pandas as pd
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import plotly.express as px
import os
# import pickle
import mlflow
import dash_bootstrap_components as dbc
from dash import Dash
#test test
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
c = 0
global EncodedLabel
EncodedLabel = {
    "Diesel": 0,
    "Petrol": 1,
    "Automatic": 0,
    "Manual": 1,
    "Dealer": [0, 0],
    "Individual": [1, 0],
    "Trustmark Dealer": [0, 1],
    "First Owner": 1,
    "Second Owner": 2,
    "Third Owner": 3,
    "Fourth & Above Owner": 4,
}

owner = ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"]
fuel = ["Diesel", "Petrol"]

seller_type = ["Individual", "Dealer", "Trustmark Dealer"]

year = list(range(1886, 2024, 1))

transmission = ["Manual", "Automatic"]

X_tran_stat = {
    "year": [2013.8134256055364, 2015.0, 2017, 4.039062067605816],
    "fuel": [0.4528719723183391, 0.0, 0, 0.4978084455072879],
    "seller_type": [0.8898269896193771, 1.0, 1, 0.3952063053306725],
    "transmission": [0.8690657439446366, 1.0, 1, 0.33735178726916865],
    "owner": [1.453287197231834, 1.0, 1, 0.707741211934925],
    "engine": [1464.542271562767, 1248.0, 1248.0, 507.5802774640794],
    "max_power": [91.86694112627987, 82.85, 74.0, 35.92656163539812],
}

y_interpret = {0 : "Low ", 1 : "Medium", 2 : "High", 3 : "Very High"}

# print(pickle.__version__)
# os.path.join(os.getcwd(), "DSAI/ML/A3/code/data/Cars.csv")
# filename = "DSAI/ML/A3/code/model/"
# filename = "model/"
# best_model = mlflow.pyfunc.load_model(os.path.join(os.getcwd(),filename))
# best_model = best_model.get_raw_model()

# loaded_model = pickle.load(open("model/car-price.model", 'rb'))
def get_latest_model():
    mlflow.set_tracking_uri("http://mlflow.ml.brain.cs.ait.ac.th/")
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'admin'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = 'password'
    os.environ['LOGNAME'] = 'st125066'
    mlflow.set_experiment(experiment_name="st125066-a3")

    # model_uri = f"models:/st125066-a3-model@dsai-ait"
    # best_model = mlflow.sklearn.load_model(model_uri)
    model_name = 'st125066-a3-model'
    stage = 'Staging'
    best_model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{stage}")
    return  best_model

best_model = get_latest_model()

# def Data_Clean(df):
#     # Remove torque feature
#     df.drop('torque',axis=1,inplace=True)

#     # Remove rows with "Test Drive Car"
#     df = df[df['owner'] != 'Test Drive Car']

#     # transform owner types to 1 - 4
#     owner = {'First Owner' : 1,
#              'Second Owner' : 2,
#              'Third Owner' : 3,
#              'Fourth & Above Owner' : 4
#              }
#     for k,v in owner.items():
#         # df['owner'].replace(k,v,inplace=True)
#         df[df['owner'] == k].owner = v

#     # Remove the rows with fuel types "LPG,CNd"
#     df = df[~df['fuel'].isin(['LPG', 'CNG'])]

#     # Replace 'kmpl' with blank and convert it to numerical
#     df.loc[df['mileage'].str.split(" ").str[1] == 'kmpl', 'mileage'] = df.loc[df['mileage'].str.split(" ").str[1] == 'kmpl', 'mileage'].str.replace(" kmpl", "")
#     df['mileage'] = df['mileage'].astype(float)

#     # Replace 'CC' with blank and convert it to numerical
#     df.loc[df['engine'].str.split(" ").str[1] == 'CC', 'engine'] = df.loc[df['engine'].str.split(" ").str[1] == 'CC', 'engine'].str.replace(" CC", "")
#     df['engine'] = df['engine'].astype(float)

#     # Replace 'bhp' with blank and convert it to numerical
#     df.loc[df['max_power'].str.split(" ").str[1] == 'bhp', 'max_power'] = df.loc[df['max_power'].str.split(" ").str[1] == 'bhp', 'max_power'].str.replace(" bhp", "")
#     df['max_power'] = df['max_power'].astype(float)

#     # rename the column and transform it into brand
#     df.rename(columns = {'name':'brand'}, inplace = True)
#     df['brand'] = df['brand'].str.split(" ").str[0]
    
#     return df
# # print(os.path.join(os.getcwd(), "DSAI/ML/A3/data/Cars.csv"))
# df = pd.read_csv(os.path.join(os.getcwd(), "data/Cars.csv"))
# # df = pd.read_csv("data/Cars.csv")


# df = Data_Clean(df)
# #x is our strong features
# X = df[['year', 'fuel', 'seller_type',
#        'transmission', 'owner', 'engine', 'max_power']]

# #y is simply selling price
# y = df["selling_price"]

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)

# X_train['engine'].fillna(X_train['engine'].median(), inplace=True)
# X_train['max_power'].fillna(X_train['max_power'].median(), inplace=True)

# from sklearn.preprocessing import StandardScaler

# # feature scaling helps improve reach convergence faster
# scaler = StandardScaler()
# X_train[['engine','max_power']] = scaler.fit_transform(X_train[['engine','max_power']])

def preprocess(v: dict):
    r = []
    for i, j in v.items():
        # value = None
        if (j == None) and (i == "engine"):
            value = scaler.transform([[j,0]])[0][0]
        elif (j == None) and (i == "max_power"):
            value = scaler.transform([[0,j]])[0][1]
        elif (j == None) and i == "seller_type":
            value = EncodedLabel["Individual"]
        elif j == None:
            value = X_tran_stat[i][2]
        else:
            if i == "year":
                value = (j - 1886) / (2024 - 1886)
            elif i in ["engine", "max_power"]:
                value = (j - X_tran_stat[i][0]) / X_tran_stat[i][3]
            else:
                value = EncodedLabel[j]
        if i == "seller_type":
            r.append(value[0])
            r.append(value[1])
        else:
            r.append(value)
    return r


# # Create a dash application
app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    dcc.Tabs(
        [
            dcc.Tab(
                label="Classification v1.0.25",
                children=[
                    html.H2(
                        children="Car Price Prediction",
                        style={
                            "textAlign": "center",
                            "color": "#503D36",
                            "font-size": 40,
                        },
                    ),
                    dbc.Stack(
                        dcc.Textarea(
                            id="explainer",
                            value="""This is machine learning model to predict categories of the selling price of the car. There are four categories which are "low","medium", "high" and "very high". There are seven features you can input. They are the year the car is produced, fuel type of the car, how many times the car is traded(owner), how the car is traded (seller type), the transmission type, the engine size measured in cc and the max power measured in bhp. After inputting, click the predict button. The output can be seen in the bottom.""",
                            style={
                                "width": "100%",
                                "height": 80,
                                "whiteSpace": "pre-line",
                            },
                            readOnly=True,
                        )
                    ),
                    dbc.Stack(
                        [
                            dcc.Dropdown(
                                id="year-dropdown",
                                options=year,
                                # value='ALL',
                                placeholder="Select model year",
                                searchable=True,
                                style={
                                    "marginRight": "10px",
                                    "margin-top": "10px",
                                    "width": "100%",
                                },
                            ),
                            dcc.Dropdown(
                                id="fuel-dropdown",
                                options=fuel,
                                # value='ALL',
                                placeholder="Select fuel type",
                                searchable=True,
                                style={
                                    "marginRight": "10px",
                                    "margin-top": "10px",
                                    "width": "100%",
                                },
                            ),
                            html.Div(
                                dcc.Dropdown(
                                    id="seller_type-dropdown",
                                    options=seller_type,
                                    # value='ALL',
                                    placeholder="Select seller type",
                                    searchable=True,
                                    style={
                                        "marginRight": "10px",
                                        "margin-top": "10px",
                                        "width": "100%",
                                    },
                                )
                            ),
                            html.Div(
                                dcc.Dropdown(
                                    id="transmission-dropdown",
                                    options=transmission,
                                    # value='ALL',
                                    placeholder="Select transmission type",
                                    searchable=True,
                                    style={
                                        "marginRight": "10px",
                                        "margin-top": "10px",
                                        "width": "100%",
                                    },
                                )
                            ),
                            html.Div(
                                dcc.Dropdown(
                                    id="owner-dropdown",
                                    options=owner,
                                    # value='ALL',
                                    placeholder="Select owner type",
                                    searchable=True,
                                    style={
                                        "marginRight": "10px",
                                        "margin-top": "10px",
                                        "width": "100%",
                                    },
                                )
                            ),
                        ],
                    ),
                    dbc.Stack(
                        [
                            # html.Br(),
                            html.Div(
                                dcc.Input(
                                    id="engine",
                                    # options=year,
                                    # value='ALL',
                                    type="number",
                                    placeholder="Enter Engine Size in CC",
                                    style={
                                        "marginRight": "10px",
                                        "margin-top": "10px",
                                        "width": "100%",
                                        "height": 30,
                                    },
                                )
                            ),
                            # html.Br(),
                            html.Div(
                                dcc.Input(
                                    id="max_power",
                                    # options=year,
                                    # value='ALL',
                                    type="number",
                                    placeholder="Enter maximum power (bhp)",
                                    style={
                                        "marginRight": "10px",
                                        "margin-top": "10px",
                                        "width": "100%",
                                        "height": 30,
                                    },
                                )
                            ),
                        ]
                    ),
                    html.Div(
                        html.Button(
                            "Predict",
                            id="predict-val",
                            n_clicks=0,
                            style={
                                "marginRight": "10px",
                                "margin-top": "10px",
                                "width": "100%",
                                "height": 50,
                                "background-color": "white",
                                "color": "black",
                            },
                        ),
                    ),
                    html.Br(),
                    dcc.Textarea(
                        id="prediction",
                        value="This is where you can see the predicted car price after clicking the predict button",
                        style={
                            "width": "100%",
                            "height": 40,
                            "whiteSpace": "pre-line",
                            "font-size": "1.5em",
                            "textAlign": "center",
                            "color": "#503D36",
                        },
                        readOnly=True,
                    ),
                ],
            ),
        ]
    )
)


@app.callback(
    Output(component_id="prediction", component_property="value"),
    [
        Input(component_id="predict-val", component_property="n_clicks"),
        #  Input(component_id='feature', component_property='children')
        # Input(component_id='feature', component_property='children'),
        Input(component_id="fuel-dropdown", component_property="value"),
        Input(component_id="seller_type-dropdown", component_property="value"),
        Input(component_id="transmission-dropdown", component_property="value"),
        Input(component_id="owner-dropdown", component_property="value"),
        Input(component_id="year-dropdown", component_property="value"),
        Input(component_id="engine", component_property="value"),
        Input(component_id="max_power", component_property="value"),
    ],
)
def predict(click, f, s, t, o, y, e, m):
    v = {
        "year": y,
        "fuel": f,
        "seller_type": s,
        "transmission": t,
        "owner": o,
        "engine": e,
        "max_power": m,
    }
    r = preprocess(v)
    sample = np.array([r])
    if click == 0:
        global c
        c = click
        result = "This is where you can see the predicted car price after clicking the predict button"
    elif click != c:
        c = click
        predicted_cat = best_model.predict(sample)[0]
        result = f"The predicted category for this car is {y_interpret[predicted_cat]}"
    else:
        result = "This is where you can see the predicted car price after clicking the predict button"
    return result


# Run the app
if __name__ == "__main__":
    app.run_server()
