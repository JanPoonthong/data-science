import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

netflix_df = pd.read_csv("../netflix_data.csv")

data_2016_to_2020 = netflix_df[(netflix_df['release_year'] >= 2016) & (netflix_df['release_year'] <= 2020)][
    ['title', 'country', 'genre', 'release_year', 'duration']]

count_country = data_2016_to_2020.groupby('country').count().sort_values(by="title", ascending=False).head(
    10).reset_index()

my_list = [1776, 434, 360, 186, 148, 147, 140, 136, 94, 74]


def func(pct, allvals):
    absolute = int(np.round(pct / 100. * np.sum(allvals)))
    return f"{pct:.1f}%\n({absolute})"


plt.pie(count_country.duration, labels=count_country.country, autopct=lambda pct: func(pct, my_list), pctdistance=0.85)
plt.show()
