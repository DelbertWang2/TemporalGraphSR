% This function calculates the normalized MSE 

function result = evaluate(SR, truthvalue, nsample)
    result.vm_mse = zeros(nsample,1);
    result.plineF_mse = zeros(nsample,1);
    for is = 1:nsample
        result.vm_mse(is) = mse(SR.vm(is,2:end,:), truthvalue.vm(is,2:end,:))/ var(reshape(truthvalue.vm(is,2:end,:),1,[]));
        result.plineF_mse(is) = mse(SR.plineF(is,2:end,:), truthvalue.plineF(is,2:end,:)) / var(reshape(truthvalue.plineF(is,2:end,:),1,[]));
        result.overall_mse(is) = mean([result.vm_mse(is); result.plineF_mse(is)]);
    end
    
    