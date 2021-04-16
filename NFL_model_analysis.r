library(ggcorrplot)
library(fpp3)
library(CustomerScoringMetrics)

Test <- as_tibble(test)
Train <- as_tibble(train)




cols <- c('result', 'compass_away', 'stadium_neutral', 'home_fav')
Test[,cols] <- lapply(Test[,cols], factor)
Train[,cols] <- lapply(Train[,cols], factor)
Train$qbelo_diff <- Train$qbelo1_pre - Train$qbelo2_pre
Test$qbelo_diff <- Test$qbelo1_pre - Test$qbelo2_pre
Train$qbvalue_diff <- Train$qb1_value_pre - Train$qb2_value_pre
Test$qbvalue_diff <- Test$qb1_value_pre - Test$qb2_value_pre



# testing for preferred model

full.model1 <- glm(result ~ stadium_neutral + log1p(dt_for_away)*compass_away + log1p(dt_for_home) + qbelo_diff + qbvalue_diff + home_fav + spread_favorite, family = binomial(), data = Train)
summary(full.model1)

reduced.model1 <- glm(result ~ stadium_neutral + log1p(dt_for_away) + compass_away + log1p(dt_for_home) + qbelo_diff + qbvalue_diff + home_fav + spread_favorite, family = binomial(), data = Train)
summary(reduced.model1)

anova(reduced.model1, full.model1, test = 'Chisq')

reduced.model2 <- glm(result ~ stadium_neutral + log1p(dt_for_away) + log1p(dt_for_home) + qbelo_diff + qbvalue_diff + home_fav + spread_favorite, family = binomial(), data = Train)
summary(reduced.model2)

anova(reduced.model2, reduced.model1, test = 'Chisq')

reduced.model3 <- glm(result ~ stadium_neutral + log1p(dt_for_away) + qbelo_diff + qbvalue_diff + home_fav + spread_favorite, family = binomial(), data = Train)
summary(reduced.model3)

anova(reduced.model3, reduced.model2, test = 'Chisq')

reduced.model4 <- glm(result ~ log1p(dt_for_away) + qbelo_diff + qbvalue_diff + home_fav + spread_favorite, family = binomial(), data = Train)
summary(reduced.model4)

anova(reduced.model4, reduced.model3, test = 'Chisq')

reduced.model5 <- glm(result ~ log1p(dt_for_away) + qbelo_diff + qbvalue_diff + home_fav, family = binomial(), data = Train)
summary(reduced.model5)

anova(reduced.model5, reduced.model4, test = 'Chisq') # previous model is preferred (reduced.model4)