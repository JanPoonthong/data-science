# Note: Mid Term Exam

Type: Lecture
Reviewed: No

Data Science for workflow

1. Data Collection and Storage -> Collect, Storage.
2. Data Preparation -> Cleaning, Organized, Reformat.
3. Exploration & Visualization -> Image, Graph.
4. Experimentation & Prediction -> Estimate, Forecast.

What do we need for machine learning?

1. A well-defined question.
2. A set of example data.
3. A new set of data to use our algorithm on.

Traditional machine learning -> Prediction, Cluster.

IOT -> Physical device.

Deep learning -> Image recognition, Language.

Data Engineer -> **Data Collection and Storage**.

- Build data flow, pipeline, storage system.
- SQL, Python, Java.

Data Analyst -> **Data Preparation and Exploration & Visualization**.

- SQL, Excel.

Data Scientist -> **Experimentation & Prediction and Data Preparation**.

- Statistical, traditional machine learning.
- Python, R.

Machine Learning Scientist -> **Prediction and Data Preparation**.

- Deep Learning, Prediction, Classification.
- Python, R.

Open data -> API, Government data, Public.

- Free data.

Company data -> Survey, customer data, logistics data, web events.

- data-driven decisions, not open data.

Quantitative data -> counted, measured, and numbers.

Qualitative data -> observed but not measured.

Unstructured –> text, video and audio files that are stored in database.

Structured –> Relational database such as MySQL.

![](img1.png)

Data pipelines -> **Transform & Load**.

- How de we keep it organized and easy to use?
  - Joining data sources into one data set.
  - Converting data structures to fit database schemas.
  - Removing irrelevant data.

Why prepare data?

- Real data is messy(Tidiness), Missing data, and Remove duplicates

Exploratory Data Analysis

- formulating hypotheses and assessing its main characteristics, with a strong emphasis on visualization.

What are experiments in data science?

- Experiments help drive decisions and draw conclusions.

1. Form a question
2. Form a hypothesis
3. Collect data
4. Test the hypothesis with statistical test
5. Interpret results

What is A/B Testing?

- Testing A case and B case and see which one produce better result.

Time series data

- Stock, gas price | Unemployment, heart, inflation rate | temperature | Height.

Forecasting time series will tell us about

- How much rainfall will we get next month?, Will traffic ease up in the next half hour?
- How will the stock market move in the next six hours?, What will be earth’s population in 20 years?

How do we know the model is good?

- Data has features and labels.

What is supervised machine learning?

- Predictions from data with labels and features.
- Recommendation systems.
- Recognizing hand-written digits

Unsupervised machine learning -> **Clustering**

- Clustering is a set of machine learning algorithms that divide data into categories, called clusters.
- Clustering can help us see patterns in messy datasets.
- Machine Learning Scientists use clustering to divide customers into segments, images into categories, or behaviors into typical and anomalous.

Histogram plot -> distribution
Scatter plot -> see two correlation between 2 subject
Line plot -> see the trend of 2 subject

```python
plt.hist(life_exp, bins=5) # for histogram plot with bins
plt.plot(x=, y=) # for line plot

# .loc[Start row: Stop row, Start column: Stop column]
brics.loc[‘BR’:'CH','country':'area']
# .iloc use index instead of string to specify row and column
brics.iloc[0:4,0:2]

plt.xscale("log")
plt.xlabel("")
plt.ylabel("")
plt.title("")

xtick_val = [1000, 10000, 100000] # for actual value
xtick_lab = ["1k", "10k", "100k"] # for text
plt.xticks(xtick_val, xtick_lab)
plt.yticks()

more_than_200 = brics['population']>=200 # output set of booleans
brics[more_than_200][['country', 'population']] # more than 2000 and showing only country and population

brics[(brics['population']>1000) | (brics['area'] < 8)][['capital']] # doing comparison and display only capital
```

![](img2.png)

```python
for lab,row in cars.iterrows():
    print(lab + ": " + str(row['cars_per_cap']))
    # lab -> index
```

Output:

![](img3.png)

```python
# Adding new row to data frame
for lab,row in cars.iterrows():
    cars.loc[lab, "COUNTRY"] = row['country'].upper()
    # "COUNTRY" -> name setting for row
```

OR

```python
cars['COUNTRY'] = cars['country'].apply(str.upper)

cars['name_length'] = cars['country'].apply(len)
```

---

```python
netflix_df.query('type == "Movie"')
netflix_df_movies_only[(netflix_df_movies_only["country"] == "United States")]

long_genre = netflix_us_only.groupby("genre")[["release_year", "duration"]].mean() # mean of release_year and duration

```

![](img4.png)

```python
x = long_genre.index # need index for plotting
plt.barh(x, long_genre.duration) # horizontal bar
```

![](img5.png)

```python
plt.scatter(long_genre.duration, x) # index on y-axis on scatter
```

![](img6.png)

```python
release_year.groupby('country').count()
```

```python
# ascending = True -> low to high -> ascending order
# ascending = False -> high to low -> descending order

release_year.groupby('country').count().sort_values(by=['title'], ascending=False).head(10)

# must be number for each variable
india = count.iloc[1][0]
uk = count.iloc[2][0]
canada = count.iloc[3][0]

y = np.array([india, canada, uk])
labels = ["India", "Canada", "United Kingdom"]

plt.pie(y, labels=labels)
```

```python
a = []
b = []
for lab, row in count.iterrows():
    a.append(row['title']) # number
    b.append(lab) # index

plt.figure(figsize=(15, 15))

def func(pct, allvals):
    absolute = int(np.round(pct / 100.0 * np.sum(allvals)))
    return f"{pct:.1f}%\n({absolute:d})"

# Pie Chart
plt.pie(a, labels=b, autopct=lambda x: func(x, a), pctdistance=0.85, startangle=90)

# draw circle
centre_circle = plt.Circle((0, 0), 0.50, fc='white')
fig = plt.gcf()

# Adding Circle in Pie chart
fig.gca().add_artist(centre_circle)

plt.title('Number of titles released by top 10 countries')
```

```python
# sorting many column
homelessness.sort_values(by=["region", "family_members"], ascending=[True, False])

mojave_state = ['Arizona', 'California', 'Nevada', "Utah"]
mojave_homelessness = homelessness[homelessness['state'].isin(mojave_state)]

homelessness['individuals'] + homelessness['family_members']
```

```python
# Dropping values
store_types = sales.drop_duplicates(subset=["store", "type"])

sales[sales['is_holiday'] == True]
holiday_dates.drop_duplicates(subset=["date"])

store_types["type"].value_counts()
store_types["type"].value_counts(normalize=True) # show in percentage of data in the data frame

store_depts["department"].value_counts(sort=True) # sort=True make into descending

temperature.set_index(["country", "city"])
temperature.reset_index() # index 0, 1, 2, n

# index two value while "country" and "city" are index
row_to_keep = [("Brazil", "Rio De Janeiro"), ("Pakistan", "Lahore")]
temperature_ind.loc[row_to_keep]
```

```python
sns.histplot(data=unemployment, x="2011", bins=20)
```