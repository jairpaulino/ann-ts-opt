# SM model

getSM = function(matrix){
  #matrix = resultsTrain
  SM = NULL
  for (i in 1:length(matrix[[1]])) { #i=1
    SM[i] = median(c(matrix[i,2], matrix[i,3], matrix[i,4]), na.rm = TRUE)
    #summary(matrix[i,2], matrix[i,3], matrix[i,4])
  }
  
  return(SM)
}

# Criar modelo RNA

getDeepEnsemble = function(resultsTrain, resultsTest){
  
  set.seed(123)
  split = sample.split(resultsTrain[[1]], SplitRatio = 0.75)
  train_set = subset(resultsTrain, split == TRUE); head(train_set)
  valid_set = subset(resultsTrain, split == FALSE); head(valid_set)
  #length(train_valid_set[[1]]) + length(test_set[[1]])
  
  test_set = resultsTest
  #length(train_set[[1]]) + length(valid_set[[1]]) + length(test_set[[1]])
  
  h2o.init(nthreads = -1)
  
  beginDE = proc.time()
  model = h2o.deeplearning(y = "obs",
                           x = names(resultsTrain)[c(2:5)],
                           training_frame = as.h2o(train_set),
                           validation_frame = as.h2o(valid_set),
                           activation = 'Tanh', 
                           #nfolds = 5,
                           replicate_training_data = TRUE,
                           hidden = c(80, 80),
                           epochs = 100, 
                           seed = 123,
                           #l1 = 1.0E-5, #ver depois
                           #l2 = 0, #ver depois
                           train_samples_per_iteration = -2
  )
  procTimeDE = proc.time() - beginDE
  
  ## sampled training data (from model building)
  h2o.performance(model, train = T)
  h2o.performance(model, valid = T)
  h2o.performance(model, newdata = as.h2o(test_set))
  
  summary(model)
  plot(model)
  
  # Predicting the Test set results
  predicoes_teste = as.data.frame(h2o.predict(model, newdata = as.h2o(test_set[-1]), rep = 10))

  result = list()
  result$forecast = predicoes_teste[[1]]
  result$procTime = procTimeDE[3]
  #rst$DE_Train = 
  #rst$DE_Test = predicoes_teste
  return(result)
}

grid = function(){
  
  # Hyper-parameter Tuning with Grid Search
  
  set.seed(123)
  split = sample.split(resultsTrain[[1]], SplitRatio = 0.75)
  train_set = subset(resultsTrain, split == TRUE); head(train_set)
  valid_set = subset(resultsTrain, split == FALSE); head(valid_set)
  #length(train_valid_set[[1]]) + length(test_set[[1]])
  
  test_set = resultsTest
  #length(train_set[[1]]) + length(valid_set[[1]]) + length(test_set[[1]])
  
  
  hyper_params <- list(
    hidden = list(c(60, 60), c(80, 80), c(100, 100), c(100, 100, 100)),
    activation = list("Tanh", "TanhWithDropout", "Rectifier", "RectifierWithDropout",
                      "Maxout", "MaxoutWithDropout"))
  hyper_params
  
  #list(strategy = "RandomDiscrete", max_models = 10, seed = 1)
  
  beginTime = proc.time()
  
  dl_grid = h2o.grid(algorithm = "deeplearning", 
                     y = "obs",
                     x = names(resultsTrain)[c(2:5)],
                     training_frame = as.h2o(train_set),
                     validation_frame = as.h2o(valid_set),
                     epochs = 100, 
                     seed = 123,
                     hyper_params = hyper_params
  )
  
  procTime = proc.time() - beginTime
  
  #write(dl_grid, file = "Results/dl_grid.txt")
  dl_grid
}
