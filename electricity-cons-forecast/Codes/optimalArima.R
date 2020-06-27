getOptimalARIMA = function(training_test){
  arima_model = auto.arima(training_test, ic = 'bic', nmodels = 5000)
  return(model)
}

getARIMAForecasts = function(test_set, model){
  onestep_arima = fitted(Arima(test_set, model = model))
  plot(onestep_arima, lwd = 2)
  lines(test_set, col = 2, lwd = 2)
  return(onestep_arima)
}

getOptimalETS = function(training_test){
  ets_model = ets(training_test, ic = 'bic')
  return(ets_model)
}

getETSForecasts = function(test_set, model){
  onestep_ets = fitted(ets(test_set, model = model))
  plot(onestep_ets, lwd = 2)
  lines(test_set, col = 2, lwd = 2)
  return(onestep_ets)
}



