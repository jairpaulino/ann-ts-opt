rm(list = ls()) #remove all data

# Importing functions
source('Codes/dataPreprocessing.R')
source('Codes/optimalArimaETS.R')
source('Codes/optimalANN.R')
source('Codes/performanceMetrics.R')

# Libraries
library(forecast) #ARIMA, ETS e NNETAR
library(neuralnet) #redes neurais
#library(GenSA)
library(GA)
library(caTools)
library(h2o)
library(pROC)

# Importar dados
data = read.csv('Data/CE_NE.csv', sep = ";"); head(data, 5)

# Phase 01 - Data Preprocessing #####
# Splitting data into training and test sets
split.data = getSplitData(data$target, training_set_size = 0.8)

# Normalizing training and test sets
normalized.data = getNormalizedData(split.data, lim_inf = 0.1, lim_sup = 0.9)

# Phase 02 - Training phase (modelling) #####
# Get optimal ARIMA, ETS and NNAR models, respectively
arima_model = getOptimalARIMA(normalized.data$training_set)
ets_model = getOptimalETS(normalized.data$training_set)
nnar_model = getOptimalNNAR(normalized.data$training_set)

# Creating result table to train set
resultsTrain = as.data.frame(matrix(nrow = length(normalized.data$training_set), 
                                   ncol = 4))
colnames(resultsTrain) = c('obs', 'ARIMA', 'ETS', 'NNAR')
resultsTrain$obs = normalized.data$training_set
resultsTrain$ARIMA = arima_model$fitted
resultsTrain$ETS = ets_model$fitted
resultsTrain$NNAR = nnar_model$fitted
#metricsTrain = getCalculatedMetrics(resultsTrain)
#View(metricsTrain)

# Get SA, SM
resultsTrain = na.omit(resultsTrain)
resultsTrain$SA = (resultsTrain$ARIMA + resultsTrain$ETS + resultsTrain$NNAR)/3
resultsTrain$SM = getSM(resultsTrain)

head(resultsTrain)
tail(resultsTrain)
# Phase 03 - Test phase (forecasting) #####
# One-step ahead approach#
onestep_arima = getARIMAForecasts(normalized.data$test_set, arima_model)
onestep_ets = getETSForecasts(normalized.data$test_set, model = ets_model)
onestep_nnar = getNNARForecasts(normalized.data$test_set, model = nnar_model)

resultsTest = as.data.frame(matrix(nrow = length(normalized.data$test_set), 
                                   ncol = 4))
colnames(resultsTest) = c('obs', 'ARIMA', 'ETS', 'NNAR')
resultsTest$obs = normalized.data$test_set
resultsTest$ARIMA = onestep_arima
resultsTest$ETS = onestep_ets
resultsTest$NNAR = onestep_nnar


# Get SA, SM and DE
resultsTest = na.omit(resultsTest)
resultsTest$SA = (resultsTest$ARIMA + resultsTest$ETS + resultsTest$NNAR)/3
resultsTest$SM = getSM(resultsTest)
resultsTest$DE = getDeepEnsemble(resultsTrain, resultsTest)

head(resultsTest); length(resultsTest[[1]])

# Phase 04 - Performance analysis #####
#Metrics

metricsTest = getCalculatedMetricsTest(resultsTest)
View(metricsTest)

# graphics
plot.ts(normalized.data$test_set, lwd = 2)
lines(onestep_arima, col = 2, lwd = 2)
lines(onestep_ets, col = 3, lwd = 2)
lines(onestep_nnar, col = 4, lwd = 2)

# getMSE(normalized.data$test_set, onestep_nnar)
# getMAPE(normalized.data$test_set, onestep_nnar)
# getARV(normalized.data$test_set, onestep_nnar)

# Optimization (in process) ####
#matriz = getAnnMatrix(normalized.data$training_set, ar = 4, ss = 10, sar = 3)
#View(matriz)
 
# set.seed(123)
# model_mlp = neuralnet(t_0 ~ .,
#                       data = matriz,
#                       learningrate = 0.05,
#                       algorithm = "rprop+",
#                       hidden = c(10, 10),
#                       rep = 10)
# plot(model_mlp)

# source('Codes/optimalANN.R')
# annParamenters = getOptGAParameters() 

# A = getAnnMatrix(ar = 2, ss = 12, sar = 3, time_series = 1:50)
# View(A)






