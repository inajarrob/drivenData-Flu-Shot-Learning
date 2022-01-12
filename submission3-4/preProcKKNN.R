library(tidyverse)
library(kknn)
#library(class)
#library(caret)

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

# eliminamos las 3 variables con mas NAs
# x_train[["income_poverty"]] = NULL
# x_train[["doctor_recc_h1n1"]] = NULL
# x_train[["doctor_recc_seasonal"]] = NULL
# x_test[["income_poverty"]] = NULL
# x_test[["doctor_recc_h1n1"]] = NULL
# x_test[["doctor_recc_seasonal"]] = NULL

# cambiamos NAs por la moda
library("DescTools")
for(i in 1:ncol(x_train)){
  x_train[,i][is.na(x_train[,i])] = Mode(x_train[,i], na.rm=TRUE)
}

# ----------------------------------------------------------------------------------------------------------------------
# dimensionality reduction
# apply MCA
# library("FactoMineR")
# library("factoextra")
# 
# aux <- x_train
# mca1 = MCA(aux, graph = FALSE)
# options(ggrepel.max.overlaps = Inf)
# res.mca = MCA(aux, graph = FALSE)
# fviz_screeplot(res.mca, addlabels = TRUE, ylim = c(0, 45))
# 
# library(PCAmixdata)
# pcaMix = PCAmix(X.quanti = NULL, X.quali = aux,rename.level = TRUE,
#          weight.col.quanti = NULL, weight.col.quali = NULL, graph = TRUE, n =7)
# summary(pcaMix)
# predict(pcaMix, X.quanti = NULL, X.quali =x_train,rename.level = TRUE)
# 
# 
# # one hot encoding
# library("mltools")
# dmy <- dummyVars(" ~ .", data = x_train)
# df_features <- data.frame(predict(dmy, newdata=x_train))
# dmy <- dummyVars(" ~ .", data = x_test)
# one_hot_test <- data.frame(predict(dmy, newdata=x_test))


# ---------------------------------------------------------------------------------------------------------------------
# preparamos tests
# todas a character
x_test <- as.data.frame(apply(x_test, 2, function(x) as_factor(x)))

# todas a factor
x_test[sapply(x_test, is.character)] = lapply(x_test[sapply(x_test, is.character)], as.factor)
respondent_id <- x_test["respondent_id"] # se guarda para la entrega
x_test <- x_test %>% select(-one_of("respondent_id"))
for(i in 1:ncol(x_test)){
  x_test[,i][is.na(x_test[,i])] = Mode(x_test[,i], na.rm=TRUE)
}

# quitar variables que aplicamos anteriormente
x_test[["health_insurance"]] = NULL
x_test[["employment_occupation"]] = NULL
x_test[["employment_industry"]] = NULL

# ---------------------------------------------------------------------------------------------------------------------
# Entrenamos con cross validation sin distancias
# ---------------------------------------------------------------------------------------------------------------------
train <- cbind(x_train, y_train)
train[["respondent_id"]] = NULL
train$y <- as.factor(train$y)
fit_knn <- train(y ~.,train[1:10000,], method="knn", preProcess= c("range"),
                 trControl=trainControl(method= 'cv', number= 10),
                 tuneGrid = expand.grid(k = c(7,9,11,13,15)))

#se predicen resultados y se escriben en un archivo para la plataforma
subida <- data.frame(respondent_id=respondent_id,
                     h1n1_vaccine=result[, 3] + result[, 4],
                     seasonal_vaccine=result[, 2] + result[, 4])
write.csv(subida, "ejemplo.csv", row.names = FALSE, quote = FALSE)

# ----------------------------------------------------------------------------------------------------------------------
#                                           Knn con distancias sin CV
# ----------------------------------------------------------------------------------------------------------------------
library(philentropy)
x_train <-  as.data.frame(lapply(x_train, function(x) as.numeric(x)))
x_test <-  as.data.frame(lapply(x_test, function(x) as.numeric(x)))
y_train <- as.data.frame(lapply(y_train, function(x) as.numeric(x)))

distance_knn_phil2 = function(train, train_labels, test=NA, k=1, metric='euclidean'){
  distances <- sapply(c(1:nrow(test)), function(x){
    sapply(c(1:nrow(train)), function(y){
      d = rbind(as.numeric(test[x,]), as.numeric(train[y,]))
      if (metric == "euclidean")
        euclidean(d[1,], d[2,], FALSE)
      else
        manhattan(d[1,], d[2,], FALSE)
    })
  })
  distances
}

res_phi <- distance_knn_phil2(x_train[1:50,], y_train[1:50,], x_test)
res_man <- distance_knn_phil2(x_train[1:50,], y_train[1:50,], x_test, metric = "manhattan")

whichpart <- function(x, n=5) {
  nx <- length(x)
  p <- nx-n
  xp <- sort(x, partial=p)[p]
  which(x > xp)
}

k=5

probsList <- sapply(c(1:nrow(res_phi)), function(x){
    clases=y_train[c(whichpart(x=res_phi[x,], n=k)), "y"] # saco las clases de los vecinos más cercanos
    ind <- as.matrix(lapply(table(clases), function(y){ y/k }))
    print(ind)
    
    finalVec <- c()
    # lista que contiene las posiciones a visitar
    index <- as.numeric(rownames(ind)) 
    i <- 1
    nextZ <- index[i]
    
    for(z in 1:4){
      if (z == nextZ){
          finalVec <- c(finalVec, ind[i])
          if (!is.na(index[i+1]))
            i <- i+1
          else
            i <- i
          nextZ <- index[i]
      } else {
          finalVec <- c(finalVec, 0)
      }
    }
    
    finalVec
})
 
prob <- t(as.data.frame(probsList, nrow=nrow(res_phi), ncol=4))
i<- 0
while(i != length(probsList)) {
  prob <- rbind(prob, c(probsList[i+1], probsList[i+2], probsList[i+3]), probsList[i+4])
  i <- i + 4
}
result <- as.matrix(prob)
colnames(result) <- c(1,2,3,4)

#se predicen resultados y se escriben en un archivo para la plataforma
subida <- data.frame(respondent_id=respondent_id,
                     h1n1_vaccine=as.numeric(result[, 3]) + as.numeric(result[, 4]),
                     seasonal_vaccine=as.numeric(result[, 2]) + as.numeric(result[, 4]))
write.csv(subida, "ejemploDist.csv", row.names = FALSE, quote = FALSE)

# ----------------------------------------------------------------------------------------------------------------------
#                                           Knn con distancia Minkowski sin CV
# ----------------------------------------------------------------------------------------------------------------------

library(kknn)
x_trainNum <-  as.data.frame(lapply(x_train, function(x) as.numeric(x)))
x_testNum <-  as.data.frame(lapply(x_test, function(x) as.numeric(x)))
y_trainNum <- as.data.frame(lapply(y_train, function(x) as.numeric(x)))
y_train <- as.data.frame(lapply(y_train, function(x) as.factor(x)))
dataNum <- cbind(x_trainNum, y_train)
dataNum["respondent_id"] = NULL
train["respondent_id"] = NULL
  
fit2 <- kknn(y ~ ., train, x_test, k = 15, distance = 2, kernel = "rectangular")
result <- fit2[["prob"]]

subida <- data.frame(respondent_id=26707:53414,
                     h1n1_vaccine=as.numeric(result[, 3]) + as.numeric(result[, 4]),
                     seasonal_vaccine=as.numeric(result[, 2]) + as.numeric(result[, 4]))
write.csv(subida, "ejemploDist.csv", row.names = FALSE, quote = FALSE)

# ----------------------------------------------------------------------------------------------------------------------
#                                  Knn con distancia Minkowski con CV y downsampling
# ----------------------------------------------------------------------------------------------------------------------

folds <- createFolds(train$y, k = 10) # me devuelve con las filas de cada fold
knnCV <- lapply(folds, function(x){
  datos=train[x,]
  #balanceo cada conjunto de training independientemente
  datos=downSample(x = datos[, -32],
                   y = datos$y)  #para meter datos sinteticos solo es poner upSample, deberia probar tambien a combinar ambas, ya que solo downsample o solo upsample no creo q sea lo ideal
  colnames(datos)[32]="y_train"
  clasificador <- kknn(y_train ~ ., datos, x_test, k = 15, distance = 2, kernel = "rectangular")
  prediccion=predict(clasificador,newdata = x_test, type='prob')
  return(prediccion)
})

salidaCV <- as.matrix(knnCV[1]$Fold01)+as.matrix(knnCV[2]$Fold02)+as.matrix(knnCV[3]$Fold03)+ as.matrix(knnCV[4]$Fold04)+as.matrix(knnCV[5]$Fold05)+as.matrix(knnCV[6]$Fold06)+as.matrix(knnCV[7]$Fold07)+as.matrix(knnCV[8]$Fold08)+as.matrix(knnCV[9]$Fold09)+as.matrix(knnCV[10]$Fold10)
sal <- as.data.frame(salidaCV/10)
salida2 = data.frame(
  respondent_id=26707:53414,
  h1n1_vaccine=sal[, 3] + sal[, 4],
  seasonal_vaccine=sal[, 2] + sal[, 4]
)

write.csv(subida, "ejemploDistCV.csv", row.names = FALSE, quote = FALSE)

