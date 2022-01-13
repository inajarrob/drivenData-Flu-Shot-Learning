library(tidyverse)
library(kknn)
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

# ---------------------------------------------------------------------------------------------------------------------
# Eliminamos las columnas que no hacen falta
# ---------------------------------------------------------------------------------------------------------------------
y_train["h1n1_vaccine"] = NULL
y_train["seasonal_vaccine"] = NULL
x_train = x_train[, -1]
#head(x_train)

# nulos
ratio_nulos <- colSums(is.na(x_train))/nrow(x_train)
as.data.frame(ratio_nulos)

# --------------------------------------------------------------------------------------------------------------------
# One hot encoding
# --------------------------------------------------------------------------------------------------------------------
# library("mltools")
# dmy <- dummyVars(" ~ .", data = x_train)
# df_features <- data.frame(predict(dmy, newdata=x_train))
# head(df_features, 1)

# ---------------------------------------------------------------------------------------------------------------------
# Eliminacion de columnas con mas NAs
# ---------------------------------------------------------------------------------------------------------------------
# x_train[["income_poverty"]] = NULL
# x_train[["doctor_recc_h1n1"]] = NULL
# x_train[["doctor_recc_seasonal"]] = NULL
# x_test[["income_poverty"]] = NULL
# x_test[["doctor_recc_h1n1"]] = NULL
# x_test[["doctor_recc_seasonal"]] = NULL

# ---------------------------------------------------------------------------------------------------------------------
# Eliminacion de filas con mas NAs
# ---------------------------------------------------------------------------------------------------------------------
# print(dim(x_train))
# porcentaje_nulos <- rowSums(is.na(x_train))/ncol(x_train)
# cutoff = 0.5
# 
# n_rows = length(which(porcentaje_nulos > cutoff))
# print("\nSe han eliminado un total de", n_rows, "filas en x_train.")
# 
# x_train = x_train[-(which(porcentaje_nulos > cutoff)),]
# y_train = y_train[-(which(porcentaje_nulos > cutoff)),]
# print(dim(x_train))

# --------------------------------------------------------------------------------------------------------------------
# Imputaciones de NAs a mano por correlaciones
# --------------------------------------------------------------------------------------------------------------------
X_train = x_train
y_train <- y_train %>% select(-one_of("respondent_id"))
Y_train = y_train
X_test = x_test

#imputacion de NAs manualmente 
X_train$employment_industry[which(is.na(X_train$employment_industry) &is.na(X_train$employment_occupation))]="D"
X_train$employment_industry[which(is.na(X_train$employment_industry) &is.na(X_train$employment_occupation))]="D"

# despues health insurance si no tienen trabajo, no tendran seguro (supongo no tener seguro 0)
X_train$health_insurance[which(X_train$employment_industry=="D" & is.na(X_train$health_insurance) )]=" 0"

# despues income_poverty si no tienen trabajo, ser?n pobres
X_train$income_poverty[which(X_train$employment_industry=="D" & is.na(X_train$income_poverty) )]="Bellow Poverty"

# si tiene mucho dinero probablemente tenga seguro
X_train$health_insurance[which(X_train$income_poverty=="> $75,000" & is.na(X_train$health_insurance) )]=" 1"

# si es d no tiene trabajo y al reves
X_train$employment_occupation[which(X_train$employment_status=="Unemployed " & is.na(X_train$employment_occupation) )]="D"
X_train$employment_industry[which(X_train$employment_status=="Unemployed " & is.na(X_train$employment_industry) )]="D"
X_train$employment_status[which(is.na(X_train$employment_status) & X_train$employment_industry=="D")]="Unemployed"

#imputo NAs manualmente 
X_test$employment_industry[which(is.na(X_test$employment_industry) &is.na(X_test$employment_occupation))]="D"
X_test$employment_industry[which(is.na(X_test$employment_industry) &is.na(X_test$employment_occupation))]="D"

# despues health insurance si no tienen trabajo, no tendran seguro (supongo no tener seguro 0)
X_test$health_insurance[which(X_test$employment_industry=="D" & is.na(X_test$health_insurance) )]=" 0"

# despues income_poverty si no tienen trabajo, ser?n pobres
X_test$income_poverty[which(X_test$employment_industry=="D" & is.na(X_test$income_poverty) )]="Bellow Poverty"

# si tiene mucho dinero probablemente tenga seguro
X_test$health_insurance[which(X_test$income_poverty=="> $75,000" & is.na(X_test$health_insurance) )]=" 1"

# si es d no tiene trabajo y al reves
X_test$employment_occupation[which(X_test$employment_status=="Unemployed " & is.na(X_test$employment_occupation) )]="D"
X_test$employment_industry[which(X_test$employment_status=="Unemployed " & is.na(X_test$employment_industry) )]="D"
X_test$employment_status[which(is.na(X_test$employment_status) & X_test$employment_industry=="D")]="Unemployed"

# --------------------------------------------------------------------------------------------------------------------
# Cambiamos NAs no cambiados por la moda
# --------------------------------------------------------------------------------------------------------------------
library("DescTools")
for(i in 1:ncol(X_train)){
  X_train[,i][is.na(X_train[,i])] = Mode(X_train[,i], na.rm=TRUE)
}

X_test <- X_test %>% select(-one_of("respondent_id"))
for(i in 1:ncol(X_test)){
  X_test[,i][is.na(X_test[,i])] = Mode(X_test[,i], na.rm=TRUE)
}

# to char
X_train <-  as.data.frame(lapply(X_train, function(x) as.character(x)))
X_test <-  as.data.frame(lapply(X_test, function(x) as.character(x)))
# to factor
X_train <-  as.data.frame(lapply(X_train, function(x) as.factor(x)))
X_test <-  as.data.frame(lapply(X_test, function(x) as.factor(x)))
# to numeric
X_train <-  as.data.frame(lapply(X_train, function(x) as.numeric(x)))
X_test <-  as.data.frame(lapply(X_test, function(x) as.numeric(x)))

# ----------------------------------------------------------------------------------------------------------------------
# dimensionality reduction
# ----------------------------------------------------------------------------------------------------------------------
# Primero vemos la varianza, en general todas tienen alta varianza
sapply(X_train, var)

# Normalizamos los datos 
df_standarized <- as.data.frame(scale(X_train))
sapply(df_standarized, sd)

# PCA
# PCA es un procedimiento matemático que utiliza la transformación ortogonal para convertir un conjunto de 
# observaciones de variables posiblemente correlacionadas en un conjunto de valores de variables linealmente
# no correlacionadas llamadas componentes principales.
train.pca <- prcomp(X_train, scale=T)
train.pca
summary(train.pca)
train.pca$sdev

dr <- data.frame(col = colnames(X_train), standardDeviation = train.pca$sdev)
Xtrain <- X_train
Xtrain <- Xtrain[-which(dr$standardDeviation < 0.8)]
Xtest <- X_test[-which(dr$standardDeviation < 0.8)]

prop_varianza <- train.pca$sdev^2 / sum(train.pca$sdev^2)
prop_varianza_acum <- cumsum(prop_varianza)
prop_varianza_acum
train.pcaTEST <- prcomp(Xtrain, scale=T)
summary(train.pcaTEST)

library(ggplot2)
ggplot(data = data.frame(prop_varianza, pc = 1:35),
       aes(x = pc, y = prop_varianza)) +
  geom_col(width = 0.3) +
  scale_y_continuous(limits = c(0,0.25)) +
  theme_bw() +
  labs(x = "Componente principal",
       y = "Prop. de varianza explicada")

ggplot(data = data.frame(prop_varianza_acum, pc = 1:35),
       aes(x = pc, y = prop_varianza_acum, group = 1)) +
  geom_point() +
  geom_line() +
  geom_label(aes(label = round(prop_varianza_acum,2))) +
  theme_bw() +
  labs(x = "Componente principal",
       y = "Prop. varianza explicada acumulada")

# --------------------------------------------------------------------------------------------------------------------
# Preparamos los datos para el entrenamiento 
# --------------------------------------------------------------------------------------------------------------------
y_train["h1n1_vaccine"] = NULL
y_train["seasonal_vaccine"] = NULL
y_train <- as.data.frame(lapply(y_train, function(x) as.factor(x)))
train <- cbind(x_train, y_train)

Y_train["h1n1_vaccine"] = NULL
Y_train["seasonal_vaccine"] = NULL
Y_train <- as.data.frame(lapply(y_train, function(x) as.factor(x)))
train <- cbind(X_train, Y_train) 
train["respondent_id"] = NULL

# ----------------------------------------------------------------------------------------------------------------------
# Knn con distancia Minkowski sin CV - Distancia EUCLIDEA
# ----------------------------------------------------------------------------------------------------------------------
#train.kknn(y ~ ., train, kmax= 30, kernel= c("triangular", "rectangular", "epanechnikov", "optimal"), distance=1)
fit1 <- kknn(y ~ ., train, Xtest, k = 22, distance = 1, kernel = "rectangular")
result <- fit1[["prob"]]

subida <- data.frame(respondent_id=26707:53414,
                     h1n1_vaccine=as.numeric(result[, 3]) + as.numeric(result[, 4]),
                     seasonal_vaccine=as.numeric(result[, 2]) + as.numeric(result[, 4]))
write.csv(subida, "ejemploDist1.csv", row.names = FALSE, quote = FALSE)

# ----------------------------------------------------------------------------------------------------------------------
# Knn con distancia Minkowski downsampling  - Distancia EUCLIDEA
# ----------------------------------------------------------------------------------------------------------------------

datos=downSample(x = train[, -35], y = train$y)  #para meter datos sinteticos solo es poner upSample, deberia probar tambien a combinar ambas, ya que solo downsample o solo upsample no creo q sea lo ideal
colnames(datos)[35]="y_train"
knnDownsample <- kknn(y_train ~ ., datos, X_test, k = 22, distance = 1, kernel = "rectangular")
resultDS <- knnDownsample[["prob"]]

subidaDS <- data.frame(respondent_id=26707:53414,
                       h1n1_vaccine=as.numeric(resultDS[, 3]) + as.numeric(resultDS[, 4]),
                       seasonal_vaccine=as.numeric(resultDS[, 2]) + as.numeric(resultDS[, 4]))

# ----------------------------------------------------------------------------------------------------------------------
# Knn con distancia Minkowski con CV y downsampling  - Distancia EUCLIDEA
# ----------------------------------------------------------------------------------------------------------------------

folds <- createFolds(train$y, k = 10) # me devuelve con las filas de cada fold
knnCV1 <- lapply(folds, function(x){
  datos=train[x,]
  #balanceo cada conjunto de training independientemente
  datos=downSample(x = datos[, -35],
                   y = datos$y)  #para meter datos sinteticos solo es poner upSample, deberia probar tambien a combinar ambas, ya que solo downsample o solo upsample no creo q sea lo ideal
  colnames(datos)[35]="y_train"
  clasificador <- kknn(y_train ~ ., datos, X_test, k = 15, distance = 1, kernel = "rectangular")
  prediccion=predict(clasificador,newdata = X_test, type='prob')
  return(prediccion)
})

salidaCV <- as.matrix(knnCV1[1]$Fold01)+as.matrix(knnCV1[2]$Fold02)+as.matrix(knnCV1[3]$Fold03)+ as.matrix(knnCV1[4]$Fold04)+as.matrix(knnCV1[5]$Fold05)+as.matrix(knnCV1[6]$Fold06)+as.matrix(knnCV1[7]$Fold07)+as.matrix(knnCV1[8]$Fold08)+as.matrix(knnCV1[9]$Fold09)+as.matrix(knnCV1[10]$Fold10)
sal <- as.data.frame(salidaCV/10)
salida2 = data.frame(
  respondent_id=26707:53414,
  h1n1_vaccine=sal[, 3] + sal[, 4],
  seasonal_vaccine=sal[, 2] + sal[, 4]
)

write.csv(salida2, "ejemploDistCV1.csv", row.names = FALSE, quote = FALSE)
