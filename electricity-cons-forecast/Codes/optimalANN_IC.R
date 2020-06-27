learningAlgorithmLabels = c("backprop", "rprop+", "rprop-", "sag", "slr")
actiovationFunctionLabels = c("logistic", "tanh")
#errorFunctionLabels = c("sse", "ce")

getAnnMatrix = function(nnParameters, series){
  sazonalityRange = nnParameters$sazonality*nnParameters$pARS
  initialIndex = sazonalityRange+1
  loss = sazonalityRange
  if(loss==0) {
    loss = nnParameters$pAR
    initialIndex = nnParameters$pAR+1
  }
  ncols = (1+nnParameters$pAR+nnParameters$pARS)
  seriesSize = length(series)
  nrows = (seriesSize-loss)
  if(nrows <=0){
    return (NULL)
  }
  else{
    ann.matrix = as.data.frame(matrix(nrow= nrows, ncol= 0))
    for(i in 0:nnParameters$pAR){
      colName_i = paste("ut_", i, sep="")
      ann.matrix[[colName_i]] = series[(initialIndex-i):(seriesSize-i)]
    }#length(series[(initialIndex-i):(seriesSize-i)])
    #View(ann.matrix)
    if(nnParameters$pARS>0){
      for(i in 1:nnParameters$pARS){
        colName_i = paste("ut_", (i*nnParameters$sazonality), sep="")
        ann.matrix[[colName_i]] = series[(initialIndex-i*nnParameters$sazonality):(seriesSize-i*nnParameters$sazonality)]
      }
    }
    #View(ann.matrix)
    return(ann.matrix)
  }
}
getAnnFormula = function(varNames){
  formula = paste(varNames[1], "~")
  varNames = varNames[-1]
  aux = paste(varNames, collapse = "+")
  formula = paste(formula, aux, sep="")
  return (formula)  
}
getNeuralNetModel = function(nnParameters, ann.matrix){
  if(is.null(ann.matrix)){
    return (NULL)
  } 
  else{
    #Neural network
    varNames = colnames(ann.matrix)
    formula = getAnnFormula(varNames)
    #set.seed(1)
    neuralNetModel=neuralnet(formula, data = ann.matrix, linear.output = F
                             , hidden = c(nnParameters$nNodes1, nnParameters$nNodes2)
                             , learningrate = nnParameters$learningRate
                             , algorithm = nnParameters$learningAlgorithm
                             , act.fct = "logistic"
                             , err.fct = "sse"
                             , threshold = 0.01, stepmax = 1e+06, rep = 1)
    return(neuralNetModel)
  }
}
getBIC <- function(nn, nnParameters){
  if(is.null(nn)){
    return (Inf)
  } else {
    residuos =  nn$data[,1]- as.numeric(nn$net.result[[1]])
    residuos = residuos[is.na(residuos)==F]
    n = length(residuos)
    LL <- function(residuos,m1,s){
      sum(stats::dnorm(residuos,m1,s,log=TRUE))
    }
    loglike = LL(residuos,mean(residuos),sd(residuos))
    if(is.na(loglike)){
      return(Inf)
    }
    nPar = ncol(nn$data)*nnParameters$nNodes1 + nnParameters$nNodes1*nnParameters$nNodes2 + nnParameters$nNodes2 + 2
    #nPar = ((nnParameters$pAR+nnParameters$pARS) + 1)*(nnParameters$nNodes+1)
    BIC = -2*(loglike) + nPar*log(n) #CORRETO
    # BIC = -2*(loglike) + 2*nPar #AIC
    # BIC = -2*(loglike) + 2*nPar +2*(nPar*(nPar+1))/(n-nPar-1) #AICC
    
    
    if(!is.finite(BIC)){
      BIC=-BIC
    }
    return(BIC)
  }
}
getOptimalANN = function(series){
  print("*** ANN Model ***")
  calls <<- 0; optimalANN <<- list()
  optimalANN$BIC = Inf
  optimalANN$model = NULL
  optimalANN$parameters = NULL
  fitness=function(parameters){# = c(2,2,12,3,.5,2)){
    calls <<- calls+1; 
    #if(calls==307){
    # g=116
    #}
    print(calls)
    print(floor(parameters))
    nnParameters = list()
    nnParameters$pAR =   floor(parameters[1])#integer in [0, 13]
    nnParameters$pARS =  floor(parameters[2])#integer in [0, 13]
    nnParameters$sazonality = floor(parameters[3])#integer in [0, 13]
    nnParameters$nNodes1 = floor(parameters[4])#integer in [1, 15]
    nnParameters$nNodes2 = floor(parameters[5])#integer in [1, 15]
    index = floor(parameters[6])#in [-.5+1e-10, 5.5-1e-10]
    nnParameters$learningAlgorithm = learningAlgorithmLabels[index]#character ("backprop", "rprop+", "rprop-", "sag", "slr")
    #index = floor(parameters[7])#in [0, 1]
    #nnParameters$actiovationFunction =  actiovationFunctionLabels[index]#character ("logistic", "tanh")
    # index = floor(parameters[7])#in [0, 1]
    # nnParameters$errorFunction =  errorFunctionLabels[index]#character ("sse", "ce")
    key = getKey(parameters, periods.keys$variablesPeriods)
    BIC = evaluatedANNs[key]
    if(is.na(BIC)){    
      nnParameters$learningRate = 0.01#parameters[7]#in[1e-10, 1-1e-10]
      ann.matrix = getAnnMatrix(nnParameters, series)#View(ann.matrix)
      neuralNetModel = getNeuralNetModel(nnParameters, ann.matrix)
      BIC = getBIC (neuralNetModel, nnParameters)
      if(BIC < optimalANN$BIC){
        optimalANN$BIC <<- BIC
        optimalANN$model <<- neuralNetModel
        optimalANN$nnParameters <<- nnParameters
        print(paste("#calls =", calls, "BIC=", BIC));
        print(paste(nnParameters));
      }
      evaluatedANNs[key] <<- BIC
    }
    return(BIC)
  }
  set.seed(0)
  n = length(series)
  max_pAR = max(2, round(.05*n))+1-1e-10; 
  max_pARS = max(1, round(.05*n))+1-1e-10;
  min_sazonality = (max_pAR+1)
  max_sazonality = max((round(.1*n)),(round(.1*n))) +1-1e-10
  max_learningAlgorithm = (length(learningAlgorithmLabels)+1-1e-5)
  max_actiovationFunctionLabels = (length(actiovationFunctionLabels)+1-1e-5)
  lowers <- c(1      , 0       , min_sazonality , 02, 02,  1                    )#,  1)#,                                             1)
  uppers <- c(max_pAR, max_pARS, max_sazonality , 15, 15,  max_learningAlgorithm)#, max_actiovationFunctionLabels)#, (length(errorFunctionLabels)+1-1e-10))
  getVariablesPeriodsAndKeysCount = function(lowers, uppers){
    dims = floor(uppers-lowers)+1
    nVars = length(dims)
    nKeys = dims[nVars]
    Pv = integer(nVars)
    Pv[nVars]=1
    for(i in (nVars-1):1){
      dim_ip1 = dims[i+1]
      Pv[i] = Pv[i+1]*dim_ip1
      nKeys = nKeys*dims[i]
    }
    ret = list(keysCount = nKeys, variablesPeriods = Pv)
    return(ret)
  }  
  periods.keys = getVariablesPeriodsAndKeysCount(lowers, uppers)
  evaluatedANNs <<- rep(NA, periods.keys$keysCount)
  getKey = function(parameters, variablesPeriods = Pv){
    zeroValues = floor(parameters) - lowers
    key = sum(zeroValues*variablesPeriods) + 1
    return(key)
  }
  tol <- 1e-3
  out <- GenSA(lower = lowers, upper = uppers, par = c(3, 2, 2, 1, 3, 1), fn = fitness 
               , control=list(max.call = 4000, max.time=300, maxit = 4000, verbose = TRUE
                              , smooth = FALSE, seed=-1, nb.stop.improvement = 40
                              , temperature = 10000))
  
  #uppers <- c(max_pAR, max_pARS, max_sazonality , 15, 15, max_learningAlgorithm)#, max_actiovationFunctionLabels)#, (length(errorFunctionLabels)+1-1e-10))
  
  #running ANN for a number of times, to take the best one
  nANNs = 1
  nnParameters = optimalANN$nnParameters
  ann.matrix = getAnnMatrix(nnParameters, series)#View(ann.matrix)
  bestArchitectureBICs = numeric(nANNs)
  for(i in 1:nANNs){    
    nnParameters$learningRate = 0.01#parameters[7]#in[1e-10, 1-1e-10]
    neuralNetModel = getNeuralNetModel(nnParameters, ann.matrix)
    BIC = getBIC (neuralNetModel, nnParameters)
    bestArchitectureBICs[i] = BIC
    if(BIC < optimalANN$BIC){
      optimalANN$BIC <<- BIC
      optimalANN$model <<- neuralNetModel
      optimalANN$nnParameters <<- nnParameters
      print(paste("#calls =", calls, "BIC = ", BIC));
      print(paste(nnParameters));
    }
  }
  
  optimal = list()
  optimal$neuralNetModel = optimalANN$model
  optimal$GenSA_output = out
  optimal$nnParameters = optimalANN$nnParameters
  optimal$evaluatedANNs = evaluatedANNs
  optimal$bestArchitectureBICs = bestArchitectureBICs
  return(optimal)
}
plotAnnArchitecture = function(nn = optAnn_i, seriesName=seriesNm_i){
  png (filename = paste(RESULTS_PATH, seriesName, ".Ann.Architecture.png", sep=""))
  plot(nn$neuralNetModel, ann = TRUE, main = seriesName, rep="best")
  dev.off()
}
getAnnForecasts = function(nn=optAnn_i, series=data.all.norm){
  # #ANN one-step forecast  (training and test set)
  #plot(series)
  annMatrix = series # in the case of wanting the ANN combined model
  if(!is.matrix(series)){#In the case of wanting the ANN single model
    annMatrix = getAnnMatrix(nn$nnParameters, series)
  }
  #View(annMatrix) 
  onestep.ann = compute(nn$neuralNetModel
                        , covariate = as.data.frame(annMatrix)[-c(1)])$net.result[,1]
  
  return (onestep.ann)
}
#optAnn = getOptimalANN(series)
