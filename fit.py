# %%
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


# %% [markdown]
# prepare data

# %%
df = pd.read_csv('data/Global_Landslide_Catalog_Export.csv')
df['event_date'] = pd.to_datetime(df['event_date'])
df['month'] = df['event_date'].dt.month
df['year'] = df['event_date'].dt.year
df.sample(3)

# %% [markdown]
# select US data from pacific coast
#

# %%
df_us = df[(df.country_code == 'US')]
df_us = df_us.query(
    "admin_division_name in ['California','Oregon', 'Washington']")

# %% [markdown]
# ### Assesing the spatial distribution of the data via KDE and saving the model

# %%


def asses_kde(df, title='',  **kde_kwargs):

    df_train, df_test = sklearn.model_selection.train_test_split(
        df, test_size=0.3)

    x = df_train['longitude']
    y = df_train['latitude']

    X = np.vstack([x, y]).T
    kde = KernelDensity(**kde_kwargs).fit(X)

    x_val = df_test['longitude']
    y_val = df_test['latitude']
    X_val = np.vstack([x_val, y_val]).T

    logprob = kde.score_samples(X_val)
    #prob = np.exp(logprob)

    fig,  ax = plt.subplots(figsize=(12, 12))
    worldmap.plot(color="lightgrey", ax=ax,)

    min_lon, min_lat, max_lon, max_lat = df['longitude'].min(
    ), df['latitude'].min(), df['longitude'].max(), df['latitude'].max()
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    color = 'b'
    cmap = 'Blues'

    ax.scatter(x, y, color=color, alpha=1, label='train')
    ax.scatter(x_val, y_val, color='k', alpha=1, marker='x', label='test')
    #sns.kdeplot(x, y, shade=True, shade_lowest=False, cmap=cmap, alpha=0.5, ax = ax, bw_adjust = 0.5)

    xgrid = np.linspace(min_lon, max_lon, 500)
    ygrid = np.linspace(min_lat, max_lat, 500)

    X, Y = np.meshgrid(xgrid, ygrid)

    Z = np.exp(kde.score_samples(np.vstack([X.ravel(), Y.ravel()]).T))
    Z = Z.reshape((500, 500))
    levels = np.linspace(0, Z.max(), 10)
    ax.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha=0.6)

    fig_kde,  ax = plt.subplots(figsize=(9, 4))
    ax.hist(logprob, bins=25)
    ax.set_title("Log Probability of test data")
    ax.set_xlabel("Log Probability")

    return kde, logprob, fig


# %%
month_dict = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
              7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

models = []
log_lkls = []
for month in range(1, 13):
    kde, logprob, fig = asses_kde(df_us.query(
        "month == @month"), title=month_dict[month], bandwidth=0.8)
    models.append(kde)
    log_lkls.append(logprob)
    pickle.dump(kde, open('model/kde_{}.pkl'.format(month_dict[month]), 'wb'))
    fig.savefig('model/kde_{}.png'.format(month_dict[month]))

# %% [markdown]
# ### Probability calibrations (a very crude way)

# %%
fig,  ax = plt.subplots(figsize=(12, 12))
bins = np.linspace(-10, -2, 50)
for i in range(12):
    ax.hist(log_lkls[i], bins=bins, alpha=0.5, label=month_dict[i+1])

ax.set_title("Log Probability of test data")
ax.set_xlabel("Log Probability")
ax.set_ylabel("Frequency")
ax.legend()
fig.savefig('model/kde_logprob.png')


threshold_dict = {'very high': -3, 'high': -4, 'medium': -
                  5, 'low': -6, 'very low': -7, 'extremely low': -10}

for key, value in threshold_dict.items():
    print(key, len(np.where(np.concatenate(log_lkls) < value)[0]))

