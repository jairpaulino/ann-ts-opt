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
#library(GA)

# Importar dados
data = read.csv('Data/CE_NE.csv', sep = ";"); head(data, 5)

# Phase 01 - Data Preprocessing #####
# Splitting data into training and test sets
split.data = getSplitData(data$target, training_set_size = 0.8)

# Normalizing training and test sets
normalized.data = getNormalizedData(split.data, lim_inf = 0.1, lim_sup = 0.9)

# Phase 02 - Training phase (modelling) #####
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
results.table = as.data.frame(matrix(nrow = length(normalized.data$test_set), 
                                     ncol = 4))
colnames(results.table) = c('test', 'ARIMA', 'ETS', 'NNAR')
results.table$test = normalized.data$test_set
results.table$ARIMA = onestep_arima
results.table$ETS = onestep_ets
results.table$NNAR = onestep_nnar

metrics.table = getCalculatedMetrics(results.table)
View(metrics.table)

# graphics
plot.ts(normalized.data$test_set, lwd = 2)
lines(onestep_arima, col = 2 , lwd = 2)
lines(onestep_ets, col = 3 , lwd = 2)
lines(onestep_nnar, col = 4 , lwd = 2)

getMSE(normalized.data$test_set, onestep_nnar)
getMAPE(normalized.data$test_set, onestep_nnar)
getARV(normalized.data$test_set, onestep_nnar)

# Optimization (in process) ####
# matriz = getAnnMatrix(normalized.data$training_set, ar = 2, ss = 12, sar = 2)
# View(matriz)
# 
# matriz.2 = cbind(normalized.data$training_set[25:391], matriz)
# names(matriz.2) = c('t_0','t_1','t_2','t_12','t_24')
# head(matriz.2)
# 
# set.seed(123)
# model_mlp = neuralnet(t_0 ~ t_1 + t_2 + t_12 + t_24 , 
#                       data = matriz.2,
#                       learningrate = 0.05,
#                       algorithm = "rprop+",
#                       hidden = c(20, 20),
#                       rep = 10
#                       )
# plot(model_mlp)



















