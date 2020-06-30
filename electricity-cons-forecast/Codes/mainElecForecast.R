rm(list = ls()) #remove all data

# Importing functions
source('Codes/dataPreprocessing.R')
source('Codes/optimalArimaETS.R')
source('Codes/optimalANN.R')
source('Codes/performanceMetrics.R')

# Libraries
library(forecast) #ARIMA, ETS e NNETAR
#library(neuralnet)
#library(GenSA)
#library(GA)

# Importar dados
data = read.csv('Data/CE_NE.csv', sep = ";"); head(data, 5)

# Phase 01 - Data Preprocessing #####
# Splitting data into training and test sets
split.data = getSplitData(data$target, training_set_size = 0.8)

# Normalizing training and test sets
normalized.data = getNormalizedData(split.data, lim_inf = 0.1, lim_sup = 0.9)

# Phase 02 - Training phase (modelling) #####
# Creating model 
# Get optimal ARIMA and ANN models, respectively
arima_model = getOptimalARIMA(normalized.data$training_set)
ets_model = getOptimalETS(normalized.data$training_set)
nnar_model = getOptimalNNAR(normalized.data$training_set)

# Phase 03 - Test phase (forecasting) #####
# One-step ahead approach#
onestep_arima = getARIMAForecasts(normalized.data$test_set, arima_model)
onestep_ets = getETSForecasts(normalized.data$test_set, model = ets_model)
onestep_nnar = getNNARForecasts(normalized.data$test_set, model = nnar_model)

# Phase 04 - Performance analysis #####

#Metrics
metrics.table = as.data.frame(matrix(nrow = 3, ncol = 3))
colnames(metrics.table) = c('ARIMA', 'ETS', 'NNAR')
rownames(metrics.table) = c('MSE', 'MAPE', 'ARV')

# Getting MSE
metrics.table$ARIMA[1] = getMSE(normalized.data$test_set, onestep_arima)

metrics.table$ETS[1] = getMSE(normalized.data$test_set, onestep_ets)

metrics.table$NNAR[1] = getMSE(normalized.data$test_set, onestep_nnar)

# Getting MAPE
metrics.table$ARIMA[2] = getMAPE(normalized.data$test_set, onestep_arima)

metrics.table$ETS[2] = getMAPE(normalized.data$test_set, onestep_ets)

metrics.table$NNAR[2] = getMAPE(normalized.data$test_set, onestep_nnar)

# Getting Theil
metrics.table$ARIMA[3] = getARV(normalized.data$test_set, onestep_arima)

metrics.table$ETS[3] = getARV(normalized.data$test_set, onestep_ets)

metrics.table$NNAR[3] = getARV(normalized.data$test_set, onestep_nnar)

View(metrics.table)

plot.ts(normalized.data$test_set, lwd = 2)
lines(onestep_arima, col = 2 , lwd = 2)
lines(onestep_ets, col = 3 , lwd = 2)
lines(onestep_nnar, col = 4 , lwd = 2)

#matriz = getAnnMatrix(normalized.data$training_set, ar = 2, ss = 12, sar = 2)
