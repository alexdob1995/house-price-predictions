# House-price-predictions

## Abstract

My main goal is to gain experience with popular tools used for Machine learning (ML).
After completing some courses online, I got the feeling that I have enough knowledge and tools to train ML models by myself.
As a beginning, I decided to try and train a classic SVM (support vector machine) model for price predictions and to spice it up I also decided to gather the Data for training by myself from legit site with real data about housing transactions that already been made in Israel.

This project includes 3 main phases:
1. Data gathering - An explanation about how I built a web-scraper and saved aroung 60000 examples.
2. Data processing - Preparing data for training and adding new features based on date and coordinates.
3. Training a prediction machine and testing the results.

## 1. Web Scraper

### Description

For training a machine that predicts prices I had to find corresponding data.
There is a website called “Madlan” (https://www.madlan.co.il/) and You can find a lot of data about Israel's housing transactions that already been made which include features
like Address details, price, size (area), number of beds etc.
Concretely, for each city there is up to 1,000 transactions.
Therefore, I built a scraper that saves the transactions data inside ".csv" tables for further purpose.

### Methodology

#### Core Libraries

1. Requests - used to get page's source-code.
2. BeautifulSoup4 - makes it easier to navigate and examine html codes.
3. Pandas - orginze and save data as csv files.

#### Pseudo
```
for city in city_list:
  1. soup = Get_source_code(url)
  2. Data = Get_data(soup)
  3. Save(Data) # using pandas
```
##### Functions
```
func Get_source_code(url):
### given url, returns soup of current url. soup is the result of html\xml text processed by beautifulsoup4 library
  1. generate random header before request. # random header contains random User-Agent string so madlan.co.il won't block me for too many requests.
  2. request the url with random generated header.
  3. wait random delay between request. # same reason as "1."
  4. check if request status is ok, if not back to step "1.".
  5. return type(soup) html code with beautifulsoup.
  
func Get_data(soup):
  1. find relevant data in soup and crop what's around. # after examining the source-code I found the data is contained in dictionary format.
  2. using json library recive a dictionary format from the text.
  3. return pandas.DataFrame the built from dictionary
```

### Obstacles encountered

"Madlan" policies: Firstly, the list of cities I found contained about 1,000 names, and after couple of requests the site blocked me and I started getting only 
bad responses. I tried to overcome this with randomizing user-agent headers and delaying between the requests but figured out that they block my requests by IP address.
To overcome this I decided to scrape top 100 most populated cities because running over 1,000 cities would take too much time and the less populated cities have poor data.

### Results

I managed to collect around 60,000 transactions. Here is an example of the data:
![Data scraped](/images/Scraped-Data-Example.png)
As you can see, some of the columns are useless and some of data is unreadable.
Handling these problems is the next step: prepare data for training the machine.

The data is not shared because it is "madlan"'s property.

## 2. Data processing

### Description

The second part of the project is to prepare the data we collected for training. If you look at the picture in the pictures folder you'll see that
there is some missing data and the address columns aren't very nice and unreadable.
In this part I'll explain how I manipulated the data and what tools I used.

### Methodology

#### Core Libraries

1. Requests - used to get addresss coordinates.
2. Pandas and numpy - fast and easier dataframe applications.
3. codecs - had rough times working with hebrew written addresss.
4. sklearn - for "feature scaling" in the end.


#### Pseudo
```
1. fix addresss - the hebrew format of the addresss made it difficult to examine and process.
Eventually, I managed to replace the "addressRecord" columns with 2 new columns: 'city', 'full_address'.
2. get address coordinates - for each sample, and each address, I got its coordinates('longtitude','latitude').
Must say thanks to https://nominatim.org/ "Open-source geocoding with OpenStreetMap data" for their API codes that got me the coordinates.
```

Here comes the creative part:
I decided to add new features for the data based on coordinates and date when the deal was made.
Coordinates: I thought about mine preferences when I'll choose a house, I want it to be close to the sea, close to the center of things, and not to far from a hospital (just in case).
Therefore after a mini-reasearch and made a list of all beaches, hospitals and downtowns. With the same algorithim that got me coordinates of addresss I got all coordinates
of these lists.
Dates: I wanted to connect between date and stock market so I found history of "Tel-Aviv Real Estate" index values and hope to find a connection between the value and the prices.

```
3. get closest distances list's of beaches, hospitals and downtowns based on coordinates.
4. replace 'date' column with "Tel-Aviv real estate index" value of current date. #Thank to tase.co.il for the data of that index
5. get rid of 'baths' column beacuse was missing too much data.
6. filled remaining missing data means of each columns based on city.
```

Now we have something like this:
![Data unscaled](/images/not-rescaled-data-for-training.png)

```
7. scaled the data with the help of sklearn in min-max approach. 
8. saved the data for training.
```

We end up with this format:
![Data rescaled](/images/rescaled-data.png)

### Obstacles encountered

Finding the coordinates of more than 60000 examples took alot of time. While applying a simple function on the whole dataframe takes a couple of seconds,
Requesting 60000 times may take a while mainly beacuse internet connection and interception of the requests.
I could split the data and run the method on several machines but after making sure the first requests went good, I simply left the computer to run at night 
and after around 10 hours it finished.

### Results

Here is the data description before rescaling:
![Data Description](/images/Data_description.png)

## 3. Training the model and scores discussion

### Description

After I collected the data and rearranged it for training, now its time to train our machine.
Watching other's similar projects I decided to give a try to scikit library and their svm methods for the training.
Unfortunatly, I didn't managed to train a decent machine and I'll discuss the results later.

### Methodology

#### Core Libraries

1. scikit - Rescaling, splitting Data and training various models.
2. pickle - saving and loading models for examination.

#### Pipeline

![Pipeline](/images/pipeline.png)

The flowchart above implemented by the code below and run several times with different kernels and different splits.

##### Code

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
import os, pickle
import pandas as pd

if __name__ == '__main__':
    os.chdir('.//Data')

    #load Data
    DF = pd.read_csv('Train_Data.csv')

    # finding best estimator base also on different splits
    score = 0   #inital score
    for test_size in [0.2,0.3,0.4,0.5,0.6]:
        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(DF.drop('amount',axis=1), DF.amount, test_size=test_size,random_state=109) # diffrent kinds of splits
        #initialing hyper_params to test and kernels
        params = [
            {'svr__C': [1, 10, 100, 1000, 10000], 'svr__kernel': ['rbf'], 'svr__gamma': [0.00001, 0.0001, 0.001, 0.01]},
        ]
        #training the model
        pipe = make_pipeline(MinMaxScaler(), SVR())
        regr = GridSearchCV(estimator=pipe, param_grid=params, n_jobs=4,cv=5)
        regr.fit(X_train, y_train)

        #updating for best model
        if score == 0 or regr.best_score_ >= score:
            score = regr.best_score_
            best_estimator = regr


    #Saving the estimator
    filename = '2nd_finalized_model.sav'
    pickle.dump(best_estimator, open(filename, 'wb'))

```

### Obstacles encountered

Training time - In my opinion it took too long for each time I run the code above. While most of the code running just 5 seconds, the .fit() method took a couple of hours each time.
Due to time consumption it was pretty hard to understand which model would be the best model because it is hard to examine how to tune the "hyper parameters" even for small subset.

### Results

Unfortunatly, when we talk about r2_score, the best I got is 0.034 by a linear model.
![Model](/images/least_bad_model.png)

If I understand correctly, That score means that the model is good as the "Mean Estimator" model. "Mean estimator" is the best estimator when the features are uncorrelated.
I tried several kernels and several parametes but every time I got similar results that look like this:
![Bad Model](/images/bad_model.png)

To see what I'm doing wrong I searched for similar projects online, most of them done pretty similar things but the exceptional thing was that these projects had much rich data.
I mean that my data had 10 features for each house and their about 60 features.

I'm not sure that this is the main problem but I learnt alot up till now and I decided to go on.

## Try it yourself

I made a simple GUI where you can enter your household properties and see if it predicts correctly. To do this follow this step:
1. Run "main.py". Make sure you have internet connection because the script send api request to get the addresss coordinates for prediction, also you should have pandas, requests and scikit-learn libraries.
2. Enter house properties and click "Submit" button.
![2nd step](/images/2step.png)
3. Don't be surprised by the predictions because the machine isn't accurate :blush: , yet some prediction are quite good.
![example](/images/example.png)

## Conclusion

I gained experience with api requests and data "scraping", also experienced preparing data for ML models and eventually training several SVM models then testing them.
Although the machine isn't working, My goal achived and now I'm heading for my next challange.
Thank you for reading this, I appreciate it and hope it wasn't a waste of time :heart_eyes: .
