func.func @foo(%X: tensor<?x?xf32>, %Y: tensor<?x?xf32>, 
        %Out: tensor<?x?xf32>) -> tensor<?x?xf32> {  
    %xx = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} 
            ins(%X, %X : tensor<?x?xf32>, tensor<?x?xf32>) 
            outs(%Out : tensor<?x?xf32>) -> tensor<?x?xf32>  
    %xy = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>} 
            ins(%X, %Y : tensor<?x?xf32>, tensor<?x?xf32>) 
            outs(%Out : tensor<?x?xf32>) -> tensor<?x?xf32>  
    %plus = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} 
            ins(%xx, %xy : tensor<?x?xf32>, tensor<?x?xf32>) 
            outs(%Out: tensor<?x?xf32>) -> tensor<?x?xf32>  
    return %plus : tensor<?x?xf32> 
}