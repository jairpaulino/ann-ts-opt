# Criar matrix a partir da ST
getAnnMatrix = function(serie_temporal, ar, ss, sar){
  
  #serie_temporal = 1:30; ar = 4; ss = 12; sar = 2
  
  matriz.sliding.window.ar = as.data.frame(matrix(nrow = length(serie_temporal), ncol = (sar+1)))
  c = 0
  for(j in  1:(ar+1)){
    for(i in 1:length(serie_temporal)){
      matriz.sliding.window.ar[(i+c),j] = serie_temporal[i]
    }  
    c = c + 1
  } #matriz.sliding.window.ar
  
  matriz.sliding.window.sar = as.data.frame(matrix(nrow = length(serie_temporal), ncol = sar))
  c = 1
  for(j in  1:(sar)){
    for(i in 1:length(serie_temporal)){
      matriz.sliding.window.sar[(i+ss*c),j] = serie_temporal[i]
    }  
    c = c + 1
  } #matriz.sliding.window.sar
  
  matriz.sliding.window = cbind(matriz.sliding.window.ar[(ss*sar + 1):length(serie_temporal),], 
                                matriz.sliding.window.sar[(ss*sar + 1):length(serie_temporal),])
  #View(matriz.sliding.window)
  return(matriz.sliding.window)
}
