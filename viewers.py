# -*- coding:utf-8 -*-

import plotly
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np 
#pio.renderers.default = "png"

"""
Available renderers:
        ['plotly_mimetype', 'jupyterlab', 'nteract', 'vscode',
         'notebook', 'notebook_connected', 'kaggle', 'azure', 'colab',
         'cocalc', 'databricks', 'json', 'png', 'jpeg', 'jpg', 'svg',
         'pdf', 'browser', 'firefox', 'chrome', 'chromium', 'iframe',
         'iframe_connected', 'sphinx_gallery']
"""


def plot_clustering(trees,is_show=True):
    fig = go.Figure()
    shapes=[]
    for i, tree in enumerate(trees):
        x = tree[:,0]
        y = tree[:,1]
        an_text = [' ']*len(x)
        an_text[2] = "Tree_{}".format(i+1),
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            text=an_text,
            mode="markers+text",
            marker=dict(
                color=i,
                colorscale="Viridis"
            )
        ))
        shapes.append(
                go.layout.Shape(            
                type="circle",
                xref="x",
                yref="y",
                x0=min(x),
                y0=min(y),
                x1=max(x),
                y1=max(y),
                opacity=0.2
                )
        )
    fig.update_layout(shapes=shapes,showlegend=False)
    fig.update_layout(yaxis=dict(scaleanchor = "x",scaleratio = 1))
    fig.update_traces(textposition='top center')
    #fig.write_image('figure.png')
    if is_show:
        fig.show()
    return fig

def plot_3d(tree,is_show=True):

    import plotly.express as px
    import pandas as pd
    df = pd.DataFrame(tree, columns=['x', 'y', 'z'])
    fig = px.scatter_3d(data_frame=df, x='x', y='y', z='z', color=None, symbol=None)
    if is_show:
        fig.show()
    return fig