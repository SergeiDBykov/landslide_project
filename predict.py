import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation
import geopandas as gpd
import sklearn
import pickle
from sklearn.neighbors import KernelDensity
pd.set_option('display.max_columns', None)
worldmap = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))


df = pd.read_csv('data/Global_Landslide_Catalog_Export.csv')
df['event_date'] = pd.to_datetime(df['event_date'])
df['month'] = df['event_date'].dt.month
df['year'] = df['event_date'].dt.year
df.sample(3)

df_us = df[(df.country_code == 'US')]
df_us = df_us.query(
    "admin_division_name in ['California','Oregon', 'Washington']")

month_dict = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
              7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}


threshold_dict = {'very high': -3, 'high': -4, 'medium': -
                  5, 'low': -6, 'very low': -7, 'extremely low': -10}


def predict(lon, lat, month: int, plot_map = True):
    model_name = './model/kde_{}.pkl'.format(month_dict[month])
    kde = pickle.load(open(model_name, 'rb'))
    X = np.vstack([lon, lat]).T
    logprob = kde.score_samples(X)
    risk = ''

    for key, value in threshold_dict.items():
        if logprob > value:
            risk = key
            break

    if plot_map:

        fig,  ax = plt.subplots(figsize=(12, 12))
        worldmap.plot(color="lightgrey", ax=ax,aspect='equal')
        df_tmp = df_us.query('month == {}'.format(month))
        x = df_tmp['longitude']
        y = df_tmp['latitude']

        min_lon, min_lat, max_lon, max_lat = df_tmp['longitude'].min(
        ), df_tmp['latitude'].min(), df_tmp['longitude'].max(), df_tmp['latitude'].max()
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        color = 'b'
        cmap = 'Blues'

        ax.scatter(x, y, color=color, alpha=1, label='historical data: '+month_dict[month])
        ax.scatter(lon, lat, color='r', alpha=1, marker='X', label='requested position, landslide risk: '+risk)
        


        xgrid = np.linspace(min_lon, max_lon, 500)
        ygrid = np.linspace(min_lat, max_lat, 500)

        X, Y = np.meshgrid(xgrid, ygrid)
        
        #Z = np.exp(kde.score_samples(np.vstack([X.ravel(), Y.ravel()]).T))
        Z = kde.score_samples(np.vstack([X.ravel(), Y.ravel()]).T)
        Z = Z.reshape((500, 500))


        #levels = [0, -3,-4,-5,-6,-7,-10]
        #level_color = ['darkred','red','orange','yellow','purple', 'green', 'lightgreen']

        #levels.reverse()
        #level_color.reverse()
        #levels = np.array(levels)

        #cts = ax.contourf(X, Y, Z, levels=levels, color = level_color, alpha = 0.6)
        
        #cts = ax.contourf(X, Y, Z, levels=[-15, -10, -7, -6, -5, -4, -3, 0], alpha = 0.6)
        cts = ax.contourf(X, Y, Z, levels=[-10, -7, -6, -5, -4, -3, 0], colors = ['lightgreen', 'green', 'blue', 'yellow', 'red', 'darkred'], alpha = 0.3)
        cbar = fig.colorbar(cts)
        cbar.ax.set_ylabel('Landslide Risk')

        #for j, lab in enumerate(['extremely low', 'very low','low','medium', 'high', 'very high']):
        #    cbar.ax.text(.5, 0.1+j/5, lab, ha='center', va='center')


        #cts.cmap.set_under('lightgreen')
        #cts.cmap.set_over('darkred')


        ax.legend()
        plt.show()
    if plot_map:
        return risk, fig
    else:
        return risk
