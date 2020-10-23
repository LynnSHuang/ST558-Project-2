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
    ##  1:25   0:52   8      :10   0:103   0:  0   0:  1      1:64       Min.   :0.1075   Min.   :0.1193  
    ##  2:27   1:52   2      : 9   1:  1   1:  0   1:103      2:33       1st Qu.:0.3463   1st Qu.:0.3412  
    ##  3:27          3      : 9           2:  0              3: 7       Median :0.5350   Median :0.5136  
    ##  4:25          5      : 9           3:104                         Mean   :0.5046   Mean   :0.4816  
    ##                6      : 9           4:  0                         3rd Qu.:0.6583   3rd Qu.:0.6122  
    ##                10     : 9           5:  0                         Max.   :0.7933   Max.   :0.7469  
    ##                (Other):49           6:  0                                                          
    ##       hum           windspeed            cnt      
    ##  Min.   :0.3600   Min.   :0.06096   Min.   : 441  
    ##  1st Qu.:0.5379   1st Qu.:0.13418   1st Qu.:2653  
    ##  Median :0.6331   Median :0.17818   Median :4642  
    ##  Mean   :0.6454   Mean   :0.18774   Mean   :4549  
    ##  3rd Qu.:0.7435   3rd Qu.:0.24360   3rd Qu.:6176  
    ##  Max.   :0.9704   Max.   :0.41543   Max.   :8173  
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

![](WednesdayAnalysis_files/figure-gfm/Explore-1.png)<!-- -->

``` r
plot(dayData$atemp)
```

![](WednesdayAnalysis_files/figure-gfm/Explore-2.png)<!-- -->

``` r
plot(dayData$hum)
```

![](WednesdayAnalysis_files/figure-gfm/Explore-3.png)<!-- -->

``` r
plot(dayData$windspeed)
```

![](WednesdayAnalysis_files/figure-gfm/Explore-4.png)<!-- -->

``` r
plot(dayData$cnt)
```

![](WednesdayAnalysis_files/figure-gfm/Explore-5.png)<!-- -->

``` r
# Do some histograms to check the distributions of numeric, non-factor variables
# Temperature, humidity variables are bimodal as expected
ggplot(data=dayData, aes(x=temp)) + geom_histogram(bins=10, aes(y=..density..)) + 
  geom_density(color="red") + labs(title="Actual Temperature")
```

![](WednesdayAnalysis_files/figure-gfm/Explore-6.png)<!-- -->

``` r
ggplot(data=dayData, aes(x=atemp)) + geom_histogram(bins=10, aes(y=..density..)) + 
  geom_density(color="red") + labs(title="Actual Temperature")
```

![](WednesdayAnalysis_files/figure-gfm/Explore-7.png)<!-- -->

``` r
ggplot(data=dayData, aes(x=hum)) + geom_histogram(bins=10, aes(y=..density..)) + 
  geom_density(color="red") + labs(title="Humidity")
```

![](WednesdayAnalysis_files/figure-gfm/Explore-8.png)<!-- -->

``` r
# Windspeed is a bit skewed right, but we're not doing linear regression. This is ok!
ggplot(data=dayData, aes(x=windspeed)) + geom_histogram(bins=10, aes(y=..density..)) + 
  geom_density(color="red") + labs(title="Windspeed")
```

![](WednesdayAnalysis_files/figure-gfm/Explore-9.png)<!-- -->

``` r
# Bike Count shows a huge spread
ggplot(data=dayData, aes(x=cnt)) + geom_histogram(bins=10, aes(y=..density..)) + 
  geom_density(color="red") + labs(title="Bike Count")
```

![](WednesdayAnalysis_files/figure-gfm/Explore-10.png)<!-- -->

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
    ##   0.00  1098.127  0.7157641   751.5930
    ##   0.01  1098.127  0.7157641   751.5930
    ##   0.02  1113.362  0.7088102   759.5823
    ##   0.03  1156.104  0.6834973   876.3273
    ##   0.04  1125.250  0.6996249   848.0137
    ##   0.05  1125.250  0.6996249   848.0137
    ##   0.06  1125.250  0.6996249   848.0137
    ##   0.07  1125.250  0.6996249   848.0137
    ##   0.08  1125.250  0.6996249   848.0137
    ##   0.09  1125.250  0.6996249   848.0137
    ##   0.10  1125.250  0.6996249   848.0137
    ##   0.11  1125.250  0.6996249   848.0137
    ##   0.12  1205.311  0.6595398   908.9019
    ##   0.13  1331.791  0.5809790  1143.4763
    ##   0.14  1381.315  0.5488703  1114.6582
    ##   0.15  1436.406  0.5163319  1141.1655
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
(unexplained variation) of 1098.1266685.

``` r
tree.cv$finalModel
```

    ## n= 72 
    ## 
    ## node), split, n, deviance, yval
    ##       * denotes terminal node
    ## 
    ##  1) root 72 300916500 4759.806  
    ##    2) yr1< 0.5 35  53917220 3295.057  
    ##      4) atemp< 0.547381 19  12148980 2335.368 *
    ##      5) atemp>=0.547381 16   3489069 4434.688 *
    ##    3) yr1>=0.5 37 100874100 6145.378  
    ##      6) temp< 0.42125 10  23267830 4144.400 *
    ##      7) temp>=0.42125 27  22737860 6886.481  
    ##       14) temp< 0.58125 11  11105020 6234.455 *
    ##       15) temp>=0.58125 16   3741201 7334.750 *

``` r
plot(tree.cv$finalModel, margin=0.2); text(tree.cv$finalModel, cex=0.8)
```

![](WednesdayAnalysis_files/figure-gfm/unnamed-chunk-1-1.png)<!-- -->

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
    ##      Min       1Q   Median       3Q      Max 
    ## -2677.00 -1157.11   -23.36  1169.32  2840.94 
    ## 
    ## Coefficients:
    ##             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)     4514       1345   3.356 0.001307 ** 
    ## temp          -14178      12176  -1.164 0.248382    
    ## windspeed      -3527       2528  -1.395 0.167566    
    ## atemp          23146      13770   1.681 0.097444 .  
    ## hum            -5040       1254  -4.018 0.000151 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 1495 on 67 degrees of freedom
    ## Multiple R-squared:  0.5025, Adjusted R-squared:  0.4728 
    ## F-statistic: 16.92 on 4 and 67 DF,  p-value: 1.238e-09

``` r
final.fit <- train(as.formula(cnt~temp+windspeed+atemp+hum),
              dayData.test,method='lm',
              trControl = trainControl(method = 'cv',number=5))
final.fit$results$RMSE
```

    ## [1] 1189.647

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

|                 |      RMSE |
| :-------------- | --------: |
| Regression Tree | 1272.7991 |
| Boosted Tree    |  695.0635 |
| linear model    | 1189.6471 |

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
