getSplitData = function(data, training_set_size=0.8){
  training_set = data[1:round(length(data)*(training_set_size))]
  test_set = data[(round((length(data)*training_set_size))+1):length(data)]
  split_data = list()
  split_data$training_set = training_set
  split_data$test_set = test_set
  return(split_data)
}

normalize = function(array, x, y){
  #Normalize to [0, 1]
  m = min(array) 
  range = max(array) - m
  norm1 = (array - m) / range
  
  #Then scale to [x,y]
  range2 = y - x
  normalized = (norm1*range2) + x
  return(normalized)
}

denormalize = function(array, min, max, x, y){
  
  range2 = y - x
  norm1 = (array-x)/range2
  return(round(norm1*(max-min)+min, 1))
  #   return(nv*(max-min)+min)
}

getNormalizedData =  function(split.data, lim_inf = 0.2, lim_sup = 0.8){
  
  training_set = normalize(split.data[[1]], lim_inf, lim_sup)
  test_set = normalize(split.data[[2]], lim_inf, lim_sup)
  
  normalized.data = list()
  normalized.data$training_set = training_set
  normalized.data$test_set = test_set
  return(normalized.data)
}

# split.data = getSplitData(AirPassengers)
# normalized.data = getNormalizedData(split.data)




