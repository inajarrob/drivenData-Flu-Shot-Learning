library(tidyverse)
library(kknn)
library(class)
library(caret)

# importar datos
x_train = read.csv("data/training_set_features.csv", na.strings = c('?','', 'NA'))
y_train = read.csv("data/training_set_labels.csv", na.strings = c('?','', 'NA'))

x_test = read.csv("data/test_set_features.csv", na.strings = c('?','', 'NA'))

submission_format = read.csv("data/submission_format.csv", na.strings = c('?','', 'NA'))

head(training_set_labels)
dim(training_set_labels)
head(training_set_features)

y_train["y"] = apply(y_train, 1, function(x){
  res = 0
  if(x["h1n1_vaccine"] == 0){
    if(x["seasonal_vaccine"] == 0){
      res = 1 # 0-0
    }else{
      res = 2 # 0-1
    }
  }else{
    if(x["seasonal_vaccine"] == 0){
      res = 3 # 1-0
    }else{
      res = 4 #1-1
    }
  }
  
  res
})


y_train["h1n1_vaccine"] = NULL
y_train["seasonal_vaccine"] = NULL
head(y_train)

# Todas a character
x_train <- as.data.frame(apply(x_train, 2, function(x) as_factor(x)))

# Todas a factor
x_train[sapply(x_train, is.character)] = lapply(x_train[sapply(x_train, is.character)], as.factor)
head(x_train)

# eliminamos las columnas que no hacen falta
x_train = x_train[, -1]
x_train[["health_insurance"]] = NULL
x_train[["employment_occupation"]] = NULL
x_train[["employment_industry"]] = NULL
head(x_train)

# nulos
ratio_nulos <- colSums(is.na(x_train))/nrow(x_train)
as.data.frame(ratio_nulos)

# preparamos tests
# todas a character
x_test <- as.data.frame(apply(x_test, 2, function(x) as_factor(x)))

# todas a factor
x_test[sapply(x_test, is.character)] = lapply(x_test[sapply(x_test, is.character)], as.factor)

respondent_id <- x_test["respondent_id"] # se guarda para la entrega
x_test <- x_test %>% select(-one_of("respondent_id"))

#quitar variables que aplicamos anteriormente
x_test[["health_insurance"]] = NULL
x_test[["employment_occupation"]] = NULL
x_test[["employment_industry"]] = NULL
train[["respondent_id"]] = NULL

# Entrenamos con cross validation
train <- cbind(x_train, y_train)
train[["respondent_id"]] = NULL
train$y <- as.factor(train$y)
fit_knn <- train(y ~.,train[1:6500,], method="knn",
                 trControl=trainControl(method="cv", number=5),
                 tuneGrid = expand.grid(k = c(7,9,11,13,15)))

#se predicen resultados y se escriben en un archivo para la plataforma
for(i in 1:ncol(x_test)){
  x_test[,i][is.na(x_test[,i])] = Mode(x_test[,i], na.rm=TRUE)
}
result <- predict(fit_knn, x_test, type = "prob")

#resultados <- predict(fit_knn, test_set_features)
subida <- data.frame(respondent_id=respondent_id,
                     h1n1_vaccine=result[, 3] + result[, 4],
                     seasonal_vaccine=result[, 2] + result[, 4])
write.csv(subida, "ejemplo.csv", row.names = FALSE, quote = FALSE)
