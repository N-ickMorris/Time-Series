# -*- coding: utf-8 -*-
"""
https://plotly.com/python/plotly-express/

@author: Nick
"""

import pandas as pd
import plotly.express as px
from plotly.offline import plot

df = pd.read_csv("election.csv")

# define the target feature and other meaningful features
targets = [" DemocratWon"]
features = ["RepublicanFraction Ohio", "RepublicanFraction Alaska", 
            "RepublicanFraction New Hampshire",
            "RepublicanFraction Arizona", "RepublicanFraction Idaho",
            "RepublicanFraction North Dakota", "RepublicanFraction Washington", 
            "RepublicanFraction Wyoming", "RepublicanFraction Wisconsin"]

# plot a matrix of scatter plots to see the whole data set
fig = px.scatter_matrix(df, 
                        dimensions=["RepublicanFraction Ohio", "RepublicanFraction Alaska", 
            "RepublicanFraction New Hampshire",
            "RepublicanFraction Arizona", "RepublicanFraction Idaho",
            "RepublicanFraction North Dakota", "RepublicanFraction Washington", 
            "RepublicanFraction Wyoming", "RepublicanFraction Wisconsin"],
                        color=" DemocratWon",
                        opacity=0.7)
fig.update_traces(diagonal_visible=False)
plot(fig)

# these 3 variables have the strongest separation for the target
group = ["RepublicanFraction Ohio", "RepublicanFraction Wyoming", "RepublicanFraction Idaho"]

# plot the group of 3 variables across categories
fig = px.scatter_3d(df, x=group[0], y=group[1], z=group[2],
                    color=" DemocratWon", opacity=0.7)
plot(fig)

# plot two variables across categories
fig = px.density_contour(df, x=group[0], y=group[1], marginal_x="histogram", 
                         marginal_y="box", color=" DemocratWon")
plot(fig)

# plot SalePrice across categories
fig = px.strip(df, y=group[0], color=" DemocratWon")
plot(fig)
