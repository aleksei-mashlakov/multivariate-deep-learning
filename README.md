# multivariate-deep-learning

### Electricity consumption

The electricity dataset is taken from https://github.com/laiguokun/multivariate-time-series-data :

"The raw dataset is in https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014. It is the electricity consumption in kWh was recorded every 15 minutes from 2011 to 2014. Because the some dimensions are equal to 0. So we eliminate the records in 2011. Final we get data contains electircity consumption of 321 clients from 2012 to 2014. And we converted the data to reflect hourly consumption."

### Open power system 

The open power system dataset is retrieved from a free and open platform for power system modeling https://open-power-system-data.org/.
The dataset corresponds to Time Series data package https://data.open-power-system-data.org/time_series/2019-06-05

The dataset represents the data originating from the European market bidding zones. This dataset contains a diverse mix of time series namely electricity consumption, market prices, wind and solar power generation with hourly resolution.
The initial dataset was preprocessed by removing the capacity and profile data, as well as series whose percentage of missing values exceeds 5\% for the defined time period. It was also limited in time for the period from January 2015 to November 2017. As a result, the data consists of 183 variables where 59 are related to load, 31 to price, 57 to wind and 36 to solar.
