# Criar matrix a partir da ST
getAnnMatrix = function(ar, ss = 12, sar = 1, time_series){
  
  #time_series = 1:30; ar = 4; ss = 12; sar = 2
  matriz.sliding.window.ar = as.data.frame(matrix(nrow = length(time_series), ncol = (ar+1)))
  c = 0
  for(j in  1:(ar+1)){
    for(i in 1:length(time_series)){
      matriz.sliding.window.ar[(i+c),j] = time_series[i]
    }  
    c = c + 1
  } #head(matriz.sliding.window.ar)
  
  names_ar = NULL
  for(i in 1:(ar+1)){
    names_ar[i] = paste("t_", (i-1), sep = "")
  }
  #names(matriz.sliding.window.ar) = names_ar
  #head(matriz.sliding.window.ar)
  
  matriz.sliding.window.sar = as.data.frame(matrix(nrow = length(time_series), ncol = sar))
  c = 1
  for(j in  1:(sar)){
    for(i in 1:length(time_series)){
      matriz.sliding.window.sar[(i+ss*c),j] = time_series[i]
    }  
    c = c + 1
  } #matriz.sliding.window.sar

  names_sar = NULL
  for(i in 1:(sar)){
    names_sar[i] = paste("t_", (ss*i), sep = "")
  }
  #names(matriz.sliding.window.sar) = names_sar
  #head(matriz.sliding.window.sar)
  
  matriz.sliding.window = cbind(matriz.sliding.window.ar[(ss*sar + 1):length(time_series),], 
                                matriz.sliding.window.sar[(ss*sar + 1):length(time_series),])
  #View(matriz.sliding.window)
  
  names(matriz.sliding.window) = c(names_ar, names_sar); head(matriz.sliding.window)
  
  return(matriz.sliding.window)
}

# Cria funcao fitness - GA
fitnessGA = function(ar, ss, sar, nh1, nh2, serie_temporal = normalized.data$training_set){
  #time_series = normalized.data$training_set; ar = 2.7; sar = 1; ss = 15.7
  
  ar = round(ar, 0); sar = round(sar, 0); ss = round(ss, 0)
  nh1 = round(nh1, 0); nh2 = round(nh2, 0)
  matriz = getAnnMatrix(normalized.data$training_set, ar = ar, ss = ss, sar = sar)
  #View(matriz)
  
  set.seed(123)
  model_mlp = neuralnet(t_0 ~ .,
                        data = matriz,
                        learningrate = 0.05,
                        algorithm = "rprop+",
                        hidden = c(nh1),#, nh2),
                        rep = 5)
  #plot(model_mlp)
  
  #length(model_mlp$net.result[[1]])
  tamanho = length(model_mlp$data[[1]])
  
  matriz.previsao = as.data.frame(matrix(nrow = tamanho, ncol = 2))
  names(matriz.previsao) = c("obs", "forecast")
  matriz.previsao$obs = model_mlp$data[[1]]
  matriz.previsao$forecast = model_mlp$net.result[[1]]
  matriz.previsao = na.omit(matriz.previsao)
  minTheil = 1/(1-getTheil(matriz.previsao$obs, matriz.previsao$forecast))
  return(minTheil)  
}

# Calcula os parametros - GA
getOptGAParameters = function(){
  #time_series = train.set; C = 0.5; n = 5.3; w = 6
  
  # c() - ar, ss, sar
  lower = c(1, 12, 01, 01, 15)
  upper = c(05, 20, 05, 01, 15)
  GA <- ga(type = "real-valued", 
           fitness =  function(x) fitnessGA (x[1], x[2], x[3], x[4], x[5]),
           lower = lower, upper = upper, 
           pcrossover = 0.95,
           pmutation = 0.2,
           popSize = 10,
           maxiter = 1000,
           parallel = T,
           run = 10,
           seed = 22)
  
  result = list()
  GA
  result$ar
  ar = round(ar, 0); sar = round(sar, 0); ss = round(ss, 0)
  nh1 = round(nh1, 0); nh2 = round(nh2, 0)
  
  
  plot(GA)
  C = summary(GA)$solution[1,][1]; n = round(summary(GA)$solution[1,][2]) 
  w = round(summary(GA)$solution[1,][3]); pos_type = round(summary(GA)$solution[1,][4]) 
  # result = c(C, n, w, pos_type)
  return(result)
}


getMLP = function(train, test){
  #train = normalized.data$training_set; head(train)
  #test = normalized.data$test_set; View(test)
  MLPTrain_df =  getAnnMatrix(ar = 11, time_series = train)
  MLPTest_df =  getAnnMatrix(ar = 11, time_series = test)
  
  #View(MLP_df)
  
  beginTrain = proc.time()
  set.seed(123)
  mlpModel = neuralnet(t_0 ~ .,
                       data = MLPTrain_df,
                       learningrate = 0.01,
                       algorithm = "rprop+",
                       act.fct = 'logistic',
                       hidden = c(10, 10),
                       rep = 5,
                       )
  procTimeTrain = proc.time() - beginTrain
  
  beginTest = proc.time()
  onestepMLP = compute(mlpModel, MLPTest_df, rep = 5)
  procTimeTest = proc.time() - beginTest
  
  result = list()
  result$train = (mlpModel$net.result[[1]] + mlpModel$net.result[[2]] + mlpModel$net.result[[3]])/3
  result$test = onestepMLP$net.result
  result$proc_time_train = procTimeTrain
  result$proc_time_test = procTimeTest
  return(result)
}
