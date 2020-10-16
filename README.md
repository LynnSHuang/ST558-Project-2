# ST558 Project 2: Bike Sharing Data Set Analysis  
Have you ever rented a bike from B-Cycle in Charlotte, NC? Then, welcome to the fascinating world of bike sharing systems. Let's examine some data from  the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset) and try to predict the amount of total rental bikes depending on predictors like time of year or time of the week.

There are n=731 observations and p=16 variables available from the Capital bike sharing system (Washington DC) in 2011-2012:

1. instant = Record index (like an observation number)
2. dteday = Date (MM/DD/YYYY format)
3. season = Categorical numeric var (1:winter, 2:spring, 3:summer, 4:fall)
4. yr = Year (0:2011, 1:2012)
5. mnth = Month (1 to 12)
6. holiday = Whether the day is a holiday or not (1/0)
7. weekday = Day of the week (0:Sunday to 6:Saturday)
8. workingday = Whether the day is a working day or weekend/holiday (1/0)
9. weathersit = Categorical numeric var for weather situation (1:mild to 4:severe)
  1. Clear, Few clouds, Partly cloudy
  2. Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
  3. Light snow, Light rain + Thunderstorm + Scattered clouds, Light rain + Scattered clouds
  4. Heavy rain + Ice pallets + Thunderstorm + Mist, Snow + Fog
10. temp = Normalized hourly temp in Celsius (Temp - Min.Temp)/(Max.Temp - Min.Temp) for Min.Temp=-8 & Max.Temp=39
11. atemp = Normalized hourly feeling temp in Celsius for Min.Temp=-16 & Max.Temp=50
12. hum = Normalized humidity (Humidity)/(Max.Humidity) for Max.Humidity=100
13. windspeed = Normalized wind speed (Speed)/(Max.Speed) for Max.Speed=67
14. casual = Count of casual users (ignored!)
15. registered = Count of registered users (ignored!)
16. cnt = Count of total rental bikes (casual + registered)


We will split the data into analyses by weekday, so 7 separate analyses for each weekday from Sunday (weekday=0) to Saturday (weekday=6). Per each weekday, we will:

* Do some preliminary numerical and graphical summaries
* Split data into 70% training, 30% test data sets
* Create a tree-based model using leave one out cross-validation
* Create a boosted tree model using cross-validation
* Comparison of model performances on the test data set, and selection of a 'best model'

## Analysis  
Here's all the analysis:  
[Sunday]()  
[Monday]()  
[Tuesday]()  
[Wednesday]()  
[Thursday]()  
[Friday]()  
[Saturday]()  


## R Markdown Automation  
Here's how to automate the analysis using the Project2.Rmd code:
1. Check that these packages are installed: 

- knitr
- rmarkdown
- tidyverse
- trees
- ?

2. Execute this code:

days <- c("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")  
outFiles <- paste0(days, "Analysis.md")  
for (i in 1:7){  
    rmarkdown::render("Project2.Rmd", output_file=outFiles[i], params=list(weekday=days[i]))  
}  

I know for-loops are a bit frowned upon in R, but this isn't bad with only 7 iterations! Plus, it's more obvious what's happening than in an apply() family function.
