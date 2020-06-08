# Credit Card Fraud Detection
# https://www.kaggle.com/mlg-ulb/creditcardfraud

require(h2o)
require(tidyverse)
require(data.table)
require(skimr)

df <- fread('../input/creditcard.csv')


df$Class <- as.factor(df$Class)

# Clustered Time...it might be useful
clusts <- kmeans(df$Time, centers = 4)
df$time2 <- clusts$cluster
df <- df %>% select(-Time)

df <- df %>% select(time2, everything())

y <- 'Class'

h2o.init()

df <- as.h2o(df)
split <- h2o.splitFrame(df, 0.7, seed=1)
train <- split[[1]]
test <- split[[2]]

h2o.table(train$Class)
h2o.table(test$Class)


aml <- h2o.automl(y=y, training_frame = train, seed=1, nfolds = 5, balance_classes = T,
                  max_runtime_secs = 600)
aml
#AUC is high at 0.9739

pred <- h2o.predict(aml, newdata = test)

perf <- h2o.performance(aml@leader, newdata = test)
perf
#AUC is at 0.9805 for test



