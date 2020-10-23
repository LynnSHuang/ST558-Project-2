ST558 Project 2: Bike Sharing Analysis
================
Lynn Huang
September 18, 2020

  - [Prepare Data](#prepare-data)
  - [Explore Data](#explore-data)
  - [Regression Tree with LOOCV](#regression-tree-with-loocv)
  - [Boosted Tree with CV](#boosted-tree-with-cv)
  - [Second Analysis](#second-analysis)
      - [Linear model](#linear-model)
          - [Final Model](#final-model)
          - [R Markdown Automation Code](#r-markdown-automation-code)

#### Prepare Data

Source: [UCI Machine Learning
Repository](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)  
We will predict the amount of total rental bikes using predictors like
time of year or time of the week.  
There are n=731 observations and p=16 variables on the Capital bike
sharing system (Washington DC) in 2011-2012:

1.  (ignored\!) instant = Record index (like an observation number)  
2.  (ignored\!) dteday = Date (MM/DD/YYYY format)  
3.  season = Categorical numeric var (1:winter, 2:spring, 3:summer,
    4:fall)  
4.  yr = Year (0:2011, 1:2012)  
5.  mnth = Month (1 to 12)  
6.  holiday = Whether the day is a holiday or not (1/0)  
7.  weekday = Day of the week (0:Sunday to 6:Saturday)  
8.  workingday = Whether the day is a working day or weekend/holiday
    (1/0)  
9.  weathersit = Categorical numeric var for weather situation (1:mild
    to 4:severe)
      - Clear, Few clouds, Partly cloudy  
      - Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist  
      - Light snow, Light rain + Thunderstorm + Scattered clouds, Light
        rain + Scattered clouds  
      - Heavy rain + Ice pallets + Thunderstorm + Mist, Snow + Fog  
10. temp = Normalized hourly temp in Celsius (Temp - Min.Temp)/(Max.Temp
    - Min.Temp) for Min.Temp=-8 & Max.Temp=39  
11. atemp = Normalized hourly feeling temp in Celsius for Min.Temp=-16 &
    Max.Temp=50  
12. hum = Normalized humidity (Humidity)/(Max.Humidity) for
    Max.Humidity=100  
13. windspeed = Normalized wind speed (Speed)/(Max.Speed) for
    Max.Speed=67  
14. (ignored\!) casual = Count of casual users  
15. (ignored\!) registered = Count of registered users  
16. cnt = Count of total rental bikes (casual + registered)

We will ignore the `casual` and `registered` variables in our analysis.
We will also split the data into analyses by weekday, so 7 separate
analyses for each weekday from Sunday (weekday=0) to Saturday
(weekday=6). This code is specifically run with \`r Per each weekday, we
will:  
\* Do some preliminary numerical and graphical summaries  
\* Split data into 70% training, 30% test data sets  
\* Create a tree-based model using leave one out cross-validation  
\* Create a boosted tree model using cross-validation  
\* Comparison of model performances on the test data set, and selection
of a ‘best model’

``` r
# Drop unused data and make factors as needed for categorical vars
bikeData <- read_csv("day.csv", col_names=TRUE) %>% select(-instant, -dteday, -casual, -registered)
```

    ## Parsed with column specification:
    ## cols(
    ##   instant = col_double(),
    ##   dteday = col_date(format = ""),
    ##   season = col_double(),
    ##   yr = col_double(),
    ##   mnth = col_double(),
    ##   holiday = col_double(),
    ##   weekday = col_double(),
    ##   workingday = col_double(),
    ##   weathersit = col_double(),
    ##   temp = col_double(),
    ##   atemp = col_double(),
    ##   hum = col_double(),
    ##   windspeed = col_double(),
    ##   casual = col_double(),
    ##   registered = col_double(),
    ##   cnt = col_double()
    ## )

``` r
bikeData$season <- as.factor(bikeData$season)
bikeData$yr <- as.factor(bikeData$yr)
bikeData$mnth <- as.factor(bikeData$mnth)
bikeData$holiday <- as.factor(bikeData$holiday)
bikeData$weekday <- as.factor(bikeData$weekday)
bikeData$workingday <- as.factor(bikeData$workingday)
bikeData$weathersit <- as.factor(bikeData$weathersit)

# Slice off data for only this weekday (default Sunday)
dayData <- bikeData %>% filter(weekday == params$day)
head(dayData)
n = nrow(dayData)

# Split into 70% training, 30% test data sets
set.seed(123)
train <- sample(1:n, size = n*0.7)
dayData.train <- dayData[train, ]
dayData.test <- dayData[-train, ]
dayData.train
```

#### Explore Data

We see an even spread across the season, yr, mnth variables (as expected
across a whole year).  
Most of the days were not a holiday. The weekday corresponds to the
report-specific day (as it should\!).

``` r
# Do some basic five-number summaries to check for outliers
summary(dayData)
```

    ##  season yr          mnth    holiday weekday workingday weathersit      temp            atemp       
    ##  1:25   0:52   3      :10   0:102   0:  0   0:  2      1:67       Min.   :0.1443   Min.   :0.1495  
    ##  2:26   1:52   5      : 9   1:  2   1:  0   1:102      2:34       1st Qu.:0.3495   1st Qu.:0.3546  
    ##  3:28          6      : 9           2:  0              3: 3       Median :0.4983   Median :0.4877  
    ##  4:25          8      : 9           3:  0                         Mean   :0.5043   Mean   :0.4827  
    ##                9      : 9           4:104                         3rd Qu.:0.6631   3rd Qu.:0.6261  
    ##                11     : 9           5:  0                         Max.   :0.8275   Max.   :0.8264  
    ##                (Other):49           6:  0                                                          
    ##       hum           windspeed            cnt      
    ##  Min.   :0.0000   Min.   :0.04727   Min.   : 431  
    ##  1st Qu.:0.5231   1st Qu.:0.13635   1st Qu.:3271  
    ##  Median :0.6029   Median :0.18345   Median :4721  
    ##  Mean   :0.6095   Mean   :0.19160   Mean   :4667  
    ##  3rd Qu.:0.7011   3rd Qu.:0.23088   3rd Qu.:6286  
    ##  Max.   :0.9396   Max.   :0.44156   Max.   :7804  
    ## 

``` r
bikeData[is.na(bikeData)==TRUE]
```

    ## <unspecified> [0]

``` r
# Take a look at the numeric, non-factor variables
# Looks like temp, atemp, cnt are clearly bimodal with 2 peaks around indices 20 (May) and 80 (July) when the weather is nice for bike riding!
plot(dayData$temp)
```

![](ThursdayAnalysis_files/figure-gfm/Explore-1.png)<!-- -->

``` r
plot(dayData$atemp)
```

![](ThursdayAnalysis_files/figure-gfm/Explore-2.png)<!-- -->

``` r
plot(dayData$hum)
```

![](ThursdayAnalysis_files/figure-gfm/Explore-3.png)<!-- -->

``` r
plot(dayData$windspeed)
```

![](ThursdayAnalysis_files/figure-gfm/Explore-4.png)<!-- -->

``` r
plot(dayData$cnt)
```

![](ThursdayAnalysis_files/figure-gfm/Explore-5.png)<!-- -->

``` r
# Do some histograms to check the distributions of numeric, non-factor variables
# Temperature, humidity variables are bimodal as expected
ggplot(data=dayData, aes(x=temp)) + geom_histogram(bins=10, aes(y=..density..)) + 
  geom_density(color="red") + labs(title="Actual Temperature")
```

![](ThursdayAnalysis_files/figure-gfm/Explore-6.png)<!-- -->

``` r
ggplot(data=dayData, aes(x=atemp)) + geom_histogram(bins=10, aes(y=..density..)) + 
  geom_density(color="red") + labs(title="Actual Temperature")
```

![](ThursdayAnalysis_files/figure-gfm/Explore-7.png)<!-- -->

``` r
ggplot(data=dayData, aes(x=hum)) + geom_histogram(bins=10, aes(y=..density..)) + 
  geom_density(color="red") + labs(title="Humidity")
```

![](ThursdayAnalysis_files/figure-gfm/Explore-8.png)<!-- -->

``` r
# Windspeed is a bit skewed right, but we're not doing linear regression. This is ok!
ggplot(data=dayData, aes(x=windspeed)) + geom_histogram(bins=10, aes(y=..density..)) + 
  geom_density(color="red") + labs(title="Windspeed")
```

![](ThursdayAnalysis_files/figure-gfm/Explore-9.png)<!-- -->

``` r
# Bike Count shows a huge spread
ggplot(data=dayData, aes(x=cnt)) + geom_histogram(bins=10, aes(y=..density..)) + 
  geom_density(color="red") + labs(title="Bike Count")
```

![](ThursdayAnalysis_files/figure-gfm/Explore-10.png)<!-- -->

#### Regression Tree with LOOCV

We will use the caret package to automate LOOCV for “rpart” method for a
regression tree.  
Because of LOOCV, this will take awhile on bigger n (we’re okay)\! Good
idea to cache results.

``` r
tree.cv <- train(cnt ~ .,
                 data = dayData.train,
                 method = "rpart",
                 trControl = trainControl(method="LOOCV"),
                 tuneGrid = expand.grid(cp=seq(0, 0.15, 0.01)))
tree.cv
```

    ## CART 
    ## 
    ## 72 samples
    ## 11 predictors
    ## 
    ## No pre-processing
    ## Resampling: Leave-One-Out Cross-Validation 
    ## Summary of sample sizes: 71, 71, 71, 71, 71, 71, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   cp    RMSE      Rsquared   MAE      
    ##   0.00  1054.337  0.7195839   842.1897
    ##   0.01  1054.337  0.7195839   842.1897
    ##   0.02  1056.897  0.7172012   848.6302
    ##   0.03  1088.813  0.7014427   870.7540
    ##   0.04  1078.616  0.7065064   861.5520
    ##   0.05  1078.616  0.7065064   861.5520
    ##   0.06  1078.616  0.7065064   861.5520
    ##   0.07  1078.616  0.7065064   861.5520
    ##   0.08  1078.616  0.7065064   861.5520
    ##   0.09  1078.616  0.7065064   861.5520
    ##   0.10  1123.245  0.6821214   883.2167
    ##   0.11  1238.696  0.6154713  1013.8218
    ##   0.12  1310.299  0.5708453  1115.2639
    ##   0.13  1249.346  0.6066244  1054.9279
    ##   0.14  1222.322  0.6207863  1028.8875
    ##   0.15  1222.322  0.6207863  1028.8875
    ## 
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final value used for the model was cp = 0.01.

``` r
best.cp <- tree.cv$bestTune$cp
best.rmse <- tree.cv$results$RMSE[tree.cv$results$cp==best.cp]
best.RSquared <- tree.cv$results$Rsquared[tree.cv$results$cp==best.cp]
best.MAE <- tree.cv$results$MAE[tree.cv$results$cp==best.cp]
```

The best complexity parameter was 0.01, based on lowest RMSE
(unexplained variation) of 1054.3367673.

``` r
tree.cv$finalModel
```

    ## n= 72 
    ## 
    ## node), split, n, deviance, yval
    ##       * denotes terminal node
    ## 
    ##  1) root 72 281959500 4788.903  
    ##    2) yr1< 0.5 35  58158620 3391.771  
    ##      4) temp< 0.3979165 9   5823062 1753.667 *
    ##      5) temp>=0.3979165 26  19825290 3958.808  
    ##       10) atemp< 0.5649365 12   8626291 3390.500 *
    ##       11) atemp>=0.5649365 14   4001305 4445.929 *
    ##    3) yr1>=0.5 37  90855480 6110.514  
    ##      6) temp< 0.4170835 10  14480690 4041.400 *
    ##      7) temp>=0.4170835 27  17706060 6876.852  
    ##       14) temp< 0.5370835 8   7140207 6213.875 *
    ##       15) temp>=0.5370835 19   5569000 7156.000 *

``` r
plot(tree.cv$finalModel, margin=0.2); text(tree.cv$finalModel, cex=0.8)
```

![](ThursdayAnalysis_files/figure-gfm/unnamed-chunk-1-1.png)<!-- -->

#### Boosted Tree with CV

We can often improve prediction using boosting, which is slow training
of trees that are grown sequentially. We make many weak, shallow trees
that each grow on a modified version of the original data, with the goal
to improve on error rate.  
Because LOOCV can be time-consuming, let’s just use 10-fold cross
validation. This could still take quite some time if you have a lot of
tuning parameters\!

``` r
# Turn warnings off because R will complain about factor levels having 0 variance
# Running this w/o tuneGrid gives n.trees=150, interaction.depth=2, shrinkage=0.1, n.minobsinnode=10
# So, try tuning in those neighborhoods of values
boost.cv <- train(cnt ~ .,
                  data = dayData.train,
                  method = "gbm",
                  distribution = "gaussian",
                  trControl = trainControl(method="cv", number=10),
                  tuneGrid = expand.grid(n.trees=c(1000, 5000, 10000),
                                         interaction.depth=1:4,
                                         shrinkage=c(0.01, 0.1),
                                         n.minobsinnode=c(1,5,10)),
                  verbose = FALSE)
boost.cv$bestTune
```

# Second Analysis

## Linear model

``` r
lm.fit <- lm(cnt~temp+windspeed+atemp+hum, data =dayData.train)
summary(lm.fit)
```

    ## 
    ## Call:
    ## lm(formula = cnt ~ temp + windspeed + atemp + hum, data = dayData.train)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -3106.5 -1254.1    31.9  1204.3  3213.7 
    ## 
    ## Coefficients:
    ##             Estimate Std. Error t value Pr(>|t|)   
    ## (Intercept)     4174       1263   3.305  0.00153 **
    ## temp           -2921      10577  -0.276  0.78324   
    ## windspeed      -3448       2484  -1.388  0.16975   
    ## atemp          10368      11847   0.875  0.38458   
    ## hum            -3914       1354  -2.892  0.00516 **
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 1574 on 67 degrees of freedom
    ## Multiple R-squared:  0.4112, Adjusted R-squared:  0.3761 
    ## F-statistic:  11.7 on 4 and 67 DF,  p-value: 2.904e-07

``` r
final.fit <- train(as.formula(cnt~temp+windspeed+atemp+hum),
              dayData.test,method='lm',
              trControl = trainControl(method = 'cv',number=5))
final.fit$results$RMSE
```

    ## [1] 1302.4

#### Final Model

We compare the final models selected from `tree.cv` ,`boost.cv` and
`linear model`for best performance on the test dataset. We measure
performance as the smallest RMSE (root mean squared error), which
reflects unexplained variation. Then, we’ll take the best model and
report it’s parameters.

``` r
regPred <- predict(tree.cv, newdata=dayData.test)
reg.rmse <- sqrt(mean((regPred - dayData.test$cnt)^2))

boostPred <- predict(boost.cv, newdata=dayData.test)
boost.rmse <- sqrt(mean((boostPred - dayData.test$cnt)^2))

RMSE.vals <- data.frame(c(reg.rmse, boost.rmse,final.fit$results$RMSE))
rownames(RMSE.vals) <- c("Regression Tree", "Boosted Tree","linear model")
colnames(RMSE.vals) <- "RMSE"
kable(RMSE.vals)
```

|                 |     RMSE |
| :-------------- | -------: |
| Regression Tree |  931.923 |
| Boosted Tree    |  894.919 |
| linear model    | 1302.400 |

We prefer the model with lower RMSE.We found that linear model is the
optimal model.

#### R Markdown Automation Code

Please don’t run this when knitting Project2.Rmd\! You’ll want to run
this separately (like in the Console) to get all the reports. Just
putting this here so it doesn’t get lost\! You can access these R
Markdwon parameters using `params$weekday` in the R Code during each
automated run.

``` r
days <- c("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
outFiles <- paste0(days, "Analysis.md")
for (i in 1:7){
  rmarkdown::render("Project2.Rmd", output_file=outFiles[i], params=list(day=(i-1)))
}
```
