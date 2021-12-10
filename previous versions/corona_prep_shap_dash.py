# import pandas as pd
# import glob
# import dash
# import dash_core_components as dcc
# import dash_html_components as html
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# import plotly.express as px
# import plotly.io as pio
# pio.renderers.default = "svg"
#
# app = dash.Dash(__name__)
#
# shap_files = [x for x in glob.glob("figure_data/*.csv") if "shap" in x]
#
# shap_dfs = [(pd.read_csv(x)
#                     .assign(subset_name = x.split("/")[1].split("_shap")[0])
#                     .drop("Unnamed: 0", axis=1)
#                    ) for x in shap_files]
# shap_df = pd.concat(shap_dfs)
#
# # tmp_shap_df = shap_df[shap_df["subset_name"] == "18 - 39"]
# # var_order = tmp_shap_df.groupby("variable").var().sort_values(by = "shap_value", ascending = False).index.tolist()
#
# fig = px.violin(shap_df,
#              x="shap_value",
#              y="variable",
#                color="actual_value",
#                facet_col="subset_name",
#                height=1000)
#
# # fig = px.strip(shap_df,
# #              x="shap_value",
# #              y="variable",
# #                color="subset_name",
# #                # facet_col="subset_name",
# #                height=1000)
#
# # fig.show()
#
# app.layout = html.Div(children=[
#     html.H1(children='Corona Preppers Shapley Values'),
#
#     html.Div(children='''
#         Shapley values grouped by important subsets
#     '''),
#
#     dcc.Graph(
#         id='example-graph',
#         figure=fig
#     )
# ])
#
# if __name__ == '__main__':
#     app.run_server(debug=True)


import dash
import dash_html_components as html
import plotly.graph_objects as go
import dash_core_components as dcc
import pandas as pd
import glob, os
import plotly.express as px
from dash.dependencies import Input, Output

colorscales = px.colors.named_colorscales()

app = dash.Dash()

shap_files = [x for x in glob.glob("figure_data/*.csv") if "shap" in x]

shap_dfs = [(pd.read_csv(x)
                    .assign(subset_name = x.split("/")[1].split("_shap")[0])
                    .drop("Unnamed: 0", axis=1)
                   ) for x in shap_files]
df = pd.concat(shap_dfs)

app.layout = html.Div(id='parent', children=[
    html.H1(id='H1', children='Coronapreppers - Catboost results', style={'textAlign': 'center', \
                                                                      'marginTop': 40, 'marginBottom': 40}),

    dcc.Dropdown(id='dropdown',
                 options=[
                     {'label': '18 - 39', 'value': '18 - 39'},
                     {'label': '40 - 59', 'value': '40 - 59'},
                     {'label': '60+', 'value': '60+'},
                    {'label': 'All', 'value': 'All'},
                     {'label': 'Lower Education', 'value': 'Lower_Education'},
                     {'label': 'Higher Education', 'value': 'Higher_Education'},
                 ],
                 value='All'),
    dcc.Graph(id='strip_plot')
])


@app.callback(Output(component_id='strip_plot', component_property='figure'),
              [Input(component_id='dropdown', component_property='value')])
def graph_update(dropdown_value):
    # print(dropdown_value)
    # fig = go.Figure([go.Scatter(x=df.loc[df["subset_name"] == '{}'.format(dropdown_value), 'shap_value'],
    #                             y=df.loc[df["subset_name"] == '{}'.format(dropdown_value), 'variable'],
    #                             # line=dict(color='firebrick', width=4)
    #                             )
    #                  ])

    fig = px.strip(df.loc[df["subset_name"] == '{}'.format(dropdown_value),: ],
                    x="shap_value",
                    y="variable",
                      color="actual_value",
                   # color_discrete_map=["#845EC2", "#D65DB1", "#FF6F91", "#FF9671", "#FFC75F", "#F9F871", "#FEFEDF"],
                   color_discrete_map={1: "#845EC2", 2: "#D65DB1", 3: "#FF6F91",
                                    4: "#FF9671", 5: "#FFC75F", 6: "#F9F871", 7: "#00C9A7"},
                   # category_orders={"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7},
                      # facet_col="subset_name",
                      height=1000)

    # fig.update_traces(marker=dict(size=8,
    #                               opacity=0.5,
    #                               line=dict(width=2,
    #                                     color='DarkSlateGrey')),
    #                   selector=dict(mode='markers'))

    fig.update_layout(title='Shapley Values Per SubGroup',
                      xaxis_title='Shapley Value',
                      yaxis_title='Questions'
                      )
    return fig


if __name__ == '__main__':
    app.run_server(host=os.getenv('IP', '0.0.0.0'),
            port=int(os.getenv('PORT', 4444)),
        debug=False)