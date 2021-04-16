library(ggcorrplot)
library(fpp3)
library(CustomerScoringMetrics)

Test <- as_tibble(test)
Train <- as_tibble(train)




cols <- c('result', 'compass_away', 'stadium_neutral', 'home_fav')
Test[,cols] <- lapply(Test[,cols], factor)
Train[,cols] <- lapply(Train[,cols], factor)

head(Test)
head(Train)
