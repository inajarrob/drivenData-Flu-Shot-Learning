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


y_train["h1n1_vaccine"] = NULL
y_train["seasonal_vaccine"] = NULL
head(y_train)


# eliminamos las columnas que no hacen falta
x_train = x_train[, -1]
x_train[["health_insurance"]] = NULL
x_train[["employment_occupation"]] = NULL
x_train[["employment_industry"]] = NULL
head(x_train)

# nulos
ratio_nulos <- colSums(is.na(x_train))/nrow(x_train)
as.data.frame(ratio_nulos)

# eliminamos las 3 variables con mas NAs
# x_train[["income_poverty"]] = NULL
# x_train[["doctor_recc_h1n1"]] = NULL
# x_train[["doctor_recc_seasonal"]] = NULL
# x_test[["income_poverty"]] = NULL
# x_test[["doctor_recc_h1n1"]] = NULL
# x_test[["doctor_recc_seasonal"]] = NULL

# ELiminar  filas con mas NAs
print(dim(x_train))
porcentaje_nulos <- rowSums(is.na(x_train))/ncol(x_train)
cutoff = 0.5

n_rows = length(which(porcentaje_nulos > cutoff))
print("\nSe han eliminado un total de", n_rows, "filas en x_train.")

x_train = x_train[-(which(porcentaje_nulos > cutoff)),]
y_train = y_train[-(which(porcentaje_nulos > cutoff)),]
print(dim(x_train))

# cambiamos NAs por la moda
library("DescTools")
for(i in 1:ncol(x_train)){
  x_train[,i][is.na(x_train[,i])] = Mode(x_train[,i], na.rm=TRUE)
}

# ---------------------------------------------------------------------------------------------------------------------
# preparamos tests
# ---------------------------------------------------------------------------------------------------------------------
x_test <- x_test %>% select(-one_of("respondent_id"))
for(i in 1:ncol(x_test)){
  x_test[,i][is.na(x_test[,i])] = Mode(x_test[,i], na.rm=TRUE)
}

# quitar variables que aplicamos anteriormente
x_test[["health_insurance"]] = NULL
x_test[["employment_occupation"]] = NULL
x_test[["employment_industry"]] = NULL

# ----------------------------------------------- CODIGO ANA Y MARCOS -------------------------------------------------
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
sum(is.na(X_train))

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

# cambiamos NAs por la moda
library("DescTools")
for(i in 1:ncol(X_train)){
  X_train[,i][is.na(X_train[,i])] = Mode(X_train[,i], na.rm=TRUE)
}

# ---------------------------------------------------------------------------------------------------------------------
# preparamos tests
# ---------------------------------------------------------------------------------------------------------------------
X_test <- X_test %>% select(-one_of("respondent_id"))
for(i in 1:ncol(X_test)){
  X_test[,i][is.na(X_test[,i])] = Mode(X_test[,i], na.rm=TRUE)
}



#######################################################
# UNDERSAMPLING PARA BALANCEAR Y_TRAIN USANDO UBTOMEK #
#######################################################

#           h1n1_vaccine    seasonal_vaccine
# Class 1:        0                0
# Class 2:        0                1
# Class 3:        1                0
# Class 4:        1                1

# Prob(h1n1_vaccine)      = Prob(Class 3) + Prob(Class 4)
# Proba(seasonal_vaccine) = Prob(Class 2) + Prob(Class 4)

# Todas a character
X_train = as.data.frame(apply(X_train, 2, function(x) as_factor(x)))
X_test = as.data.frame(apply(X_test, 2, function(x) as_factor(x)))

# Todas a factor
X_train[sapply(X_train, is.character)] = lapply(X_train[sapply(X_train, is.character)], as.factor)
X_test[sapply(X_test, is.character)] = lapply(X_test[sapply(X_test, is.character)], as.factor)
# head(X_train)

# Todas a numerico
X_train[sapply(X_train, is.factor)] = lapply(X_train[sapply(X_train, is.factor)], as.numeric)
X_test[sapply(X_test, is.factor)] = lapply(X_test[sapply(X_test, is.factor)], as.numeric)
#head(X_train)

library(DescTools)
for(i in 1:ncol(X_train)){
  X_train[,i][is.na(X_train[,i])] = Mode(X_train[,i], na.rm=TRUE)
}

for(i in 1:ncol(X_test)){
  X_test[,i][is.na(X_test[,i])] = Mode(X_test[,i], na.rm=TRUE)
}


print("y_train")
print(unique(Y_train$y))
print(dim(Y_train))

Y_train <- as.data.frame(lapply(y_train, function(x) as.numeric(x)))


##### y_h1n1
Y_train$y_h1n1 = Y_train$y
# (1, 2) --> 0 --> no tiene la vacuna h1n1
Y_train = Y_train %>% mutate(y_h1n1=replace(y_h1n1, y_h1n1 %in% c(1, 2), 0))

# (3, 4) --> 1 --> sí tiene la vacuna h1n1
Y_train = Y_train %>% mutate(y_h1n1=replace(y_h1n1, y_h1n1 %in% c(3, 4), 1))

Y_train$y_h1n1 = as_factor(Y_train$y_h1n1)
length(Y_train$y_h1n1)

##### y_seasonal
Y_train$y_seasonal = Y_train$y
# (1, 3) --> 0 --> no tiene la vacuna seasonal
Y_train = Y_train %>% mutate(y_seasonal=replace(y_seasonal, y_seasonal %in% c(1, 3), 0))

# (2, 4) --> 1 --> sí tiene la vacuna seasonal
Y_train = Y_train %>% mutate(y_seasonal=replace(y_seasonal, y_seasonal %in% c(2, 4), 1))

Y_train$y_seasonal = as_factor(Y_train$y_seasonal)
length(Y_train$y_seasonal)

# Tomek links are pairs of instances of opposite classes who are their own nearest neighbors. 
# In other words, they are pairs of opposing instances that are very close together.
# Tomek's algorithm looks for such pairs and removes the majority instance of the pair.
library('unbalanced')

# Número de celdas con NAs
sum(is.na(X_train))
sum(is.na(Y_train))

data_h1n1 = ubTomek(X=X_train, Y=Y_train$y_h1n1, verbose = TRUE)
X_train_h1n1 = data_h1n1$X
Y_train_h1n1 = data_h1n1$Y

data_seasonal = ubTomek(X=X_train, Y=Y_train$y_seasonal, verbose = TRUE)
X_train_seasonal = data_seasonal$X
Y_train_seasonal = data_seasonal$Y
#necesario para usar ubTomek que solo haya 2 clases en la variable y

# clasificamos 
train <- cbind(X_train_h1n1, Y_train_h1n1)

# ----------------------------------------------------------------------------------------------------------------------
# Knn con distancia Minkowski sin CV - Distancia 1 -> EUCLIDEAN
# ----------------------------------------------------------------------------------------------------------------------
y_train["h1n1_vaccine"] = NULL
y_train["seasonal_vaccine"] = NULL
y_train <- as.data.frame(lapply(y_train, function(x) as.factor(x)))
train <- cbind(x_train, y_train)

Y_train["h1n1_vaccine"] = NULL
Y_train["seasonal_vaccine"] = NULL
Y_train <- as.data.frame(lapply(y_train, function(x) as.factor(x)))
train <- cbind(X_train, Y_train) # codigo ana
train["respondent_id"] = NULL

train.kknn(y ~ ., train, kmax= 30, kernel= c("triangular", "rectangular", "epanechnikov", "optimal"), distance=1)
fit1 <- kknn(y ~ ., train, X_test, k = 22, distance = 1, kernel = "rectangular")
#fitUbtomek <- kknn(Y_train_h1n1 ~ ., train, X_test, k = 27, distance = 1, kernel = "rectangular")
resultNAhand <- fit1[["prob"]]
#resultUbtomek <- fitUbtomek[["prob"]]

subida <- data.frame(respondent_id=26707:53414,
                     h1n1_vaccine=as.numeric(resultNAhand[, 3]) + as.numeric(resultNAhand[, 4]),
                     seasonal_vaccine=as.numeric(resultNAhand[, 2]) + as.numeric(resultNAhand[, 4]))
# subidaUb <- data.frame(respondent_id=26707:53414,
#                      h1n1_vaccine=as.numeric(resultUbtomek[, 1]),
#                      seasonal_vaccine=as.numeric(resultUbtomek[, 2]))
write.csv(subida, "ejemploDist1NAhand.csv", row.names = FALSE, quote = FALSE)

# ----------------------------------------------------------------------------------------------------------------------
# Knn con distancia Minkowski downsampling  - Distancia 1
# ----------------------------------------------------------------------------------------------------------------------

datos=downSample(x = train[, -33], y = train$y)  #para meter datos sinteticos solo es poner upSample, deberia probar tambien a combinar ambas, ya que solo downsample o solo upsample no creo q sea lo ideal
colnames(datos)[33]="y_train"
knnDownsample <- kknn(y_train ~ ., datos, x_test, k = 15, distance = 1, kernel = "rectangular")
resultDS <- knnDownsample[["prob"]]

subidaDS <- data.frame(respondent_id=26707:53414,
                       h1n1_vaccine=as.numeric(resultDS[, 3]) + as.numeric(resultDS[, 4]),
                       seasonal_vaccine=as.numeric(resultDS[, 2]) + as.numeric(resultDS[, 4]))

# ----------------------------------------------------------------------------------------------------------------------
# Knn con distancia Minkowski con CV y downsampling  - Distancia 1
# ----------------------------------------------------------------------------------------------------------------------

folds <- createFolds(train$y, k = 10) # me devuelve con las filas de cada fold
knnCV1 <- lapply(folds, function(x){
  datos=train[x,]
  #balanceo cada conjunto de training independientemente
  datos=downSample(x = datos[, -33],
                   y = datos$y)  #para meter datos sinteticos solo es poner upSample, deberia probar tambien a combinar ambas, ya que solo downsample o solo upsample no creo q sea lo ideal
  colnames(datos)[33]="y_train"
  clasificador <- kknn(y_train ~ ., datos, x_test, k = 15, distance = 1, kernel = "rectangular")
  prediccion=predict(clasificador,newdata = x_test, type='prob')
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
