rm(list = ls())
#setwd('C:/Users/jairp/Dropbox/_Papers_Books/01_Artigos_em_andamento/Time series/')
getwd()
# Importing functions
source('Codes/dataPreprocessing.R')
source('Codes/optimalArimaETS.R')
source('Codes/optimalANN.R')
source('Codes/performanceMetrics.R')

# Libraries
library(forecast)
library(neuralnet)
library(GenSA)

# Importing data
data = read.csv('Data/CEL_NE.csv', sep = ";")

# Data Preprocessing #####
# Splitting data into training and test sets
split.data = getSplitData(data$target, training_set_size = 0.8)

# Normalizing training and test sets
normalized.data = getNormalizedData(split.data, lim_inf = 0.1, lim_sup = 0.9)

# Creating model #####
# Get optimal ARIMA and ANN models, respectively
arima_model = auto.arima(normalized.data$training_set, ic = 'bic', nmodels = 5000)
ann_model = getOptimalANN(normalized.data$training_set)
ets_model = getOptimalETS(normalized.data$training_set)

# One-step ahead approach #####
# ARIMA
onestep_arima = fitted(Arima(normalized.data$test_set, model = arima_model))

# ANN
#complete_data = c(normalized.data$training_set, normalized.data$test_set)  
onestep_ann = getAnnForecasts(ann_model, normalized.data$test_set)
#onestep_ann = onestep_ann[(length(onestep_ann)-length(normalized.data$test_set)+1):length(onestep_ann)]

# ETS
onestep_ets = getETSForecasts(normalized.data$test_set, model = ets_model)

# ANN Parameters
ann_model$nnParameters

# Plot
#length(normalized.data$test_set); length(onestep_ann)
plot_size_start = length(normalized.data$test_set) - length(onestep_ann)
plot_size_end = length(normalized.data$test_set)
test_set_onestep = normalized.data$test_set[(plot_size_start+1):plot_size_end]

plot.ts(test_set_onestep, lwd = 2)
lines(onestep_ann, col = 2, lwd = 2)

#Metrics
metrics.table = as.data.frame(matrix(nrow = 3, ncol = 3))
colnames(metrics.table) = c('ARIMA', 'ETS', 'ANN')
rownames(metrics.table) = c('MSE', 'MAPE', 'Theil')

# Getting MSE
metrics.table$ARIMA[1] = getMSE(test_set_onestep, 
                                onestep_arima[(plot_size_start+1):plot_size_end])

metrics.table$ETS[1] = getMSE(test_set_onestep, 
                              onestep_ets[(plot_size_start+1):plot_size_end])

metrics.table$ANN[1] = getMSE(test_set_onestep, onestep_ann)



# Getting MAPE
metrics.table$ARIMA[2] = getMAPE(test_set_onestep, 
                                 onestep_arima[(plot_size_start+1):plot_size_end])

metrics.table$ETS[2] = getMAPE(test_set_onestep, 
                               onestep_ets[(plot_size_start+1):plot_size_end])

metrics.table$ANN[2] = getMAPE(test_set_onestep, onestep_ann)

# Getting Theil
metrics.table$ARIMA[3] = getTheil(test_set_onestep, 
                                  onestep_arima[(plot_size_start+1):plot_size_end])

metrics.table$ETS[3] = getTheil(test_set_onestep, 
                                onestep_ets[(plot_size_start+1):plot_size_end])

metrics.table$ANN[3] = getTheil(test_set_onestep, onestep_ann)

View(metrics.table)

plot.ts(test_set_onestep, lwd = 2)
lines(onestep_arima[(plot_size_start+1):plot_size_end], col = 2 , lwd = 2)
lines(onestep_ets[(plot_size_start+1):plot_size_end], col = 3 , lwd = 2)
lines(onestep_ann, col = 4 , lwd = 2)

matriz = getAnnMatrix(1:50, ar = 5, ss = 10, sar = 3)
