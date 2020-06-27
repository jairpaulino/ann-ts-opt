getMSE = function(target,forecast){
  MSE=(sum((target-forecast)^2))/length(target)
  MSE
}
getMAPE = function(target,forecast){
  target[which(target==0)] = 1e-10*min(abs(target[target!=0]))
  MAPE=sum(abs((target-forecast)/target))/length(target)
  MAPE  
}
getTheil = function(target,forecast){
  i=1
  seriesSize = length(target)
  squaredSumTF = 0
  squaredSumTT = 0
  while(i<seriesSize){
    squaredSumTF = squaredSumTF + (target[i]-forecast[i])^2
    squaredSumTT = squaredSumTT + (target[i]-target[i+1])^2
    #valor.theil[i]=((target[i]-forecast[i])^2)/(sum((target[i]-target[i+1])^2))
    i=i+1
  }
  Theil = squaredSumTF/squaredSumTT
  Theil
}
getARV = function(target,forecast){
  meanT = mean(target)
  ARV=(sum((target-forecast)^2))/(sum((forecast-meanT)^2))
  ARV
}
getPOCID = function(target,forecast){
  Dt = 0
  i=1
  seriesSize = length(target)
  while(i<seriesSize){
    TT = (target[i+1]-target[i])
    FF = (forecast[i+1]-forecast[i])
    #if((target[i+1]-target[i])*(forecast[i+1]-forecast[i])>0){
    if(TT*FF>0 | (TT==0 & FF==0)){
      Dt= Dt + 1 
    }
    i=i+1
    #print(i)
  }
  POCID=100*(Dt/seriesSize)
  POCID
}
#WRONG Prediction on Change of Direction
getWPOCID = function(target,forecast){
  Dt = 0
  i=1
  seriesSize = length(target)
  while(i<seriesSize){
    TT = (target[i+1]-target[i])
    FF = (forecast[i+1]-forecast[i])
    #if((target[i+1]-target[i])*(forecast[i+1]-forecast[i])>0){
    if(TT*FF>0 | (TT==0 & FF==0)){
      Dt= Dt + 1 
    }
    i=i+1
    #print(i)
  }
  POCID_=1-(Dt/seriesSize)
  POCID_
}
