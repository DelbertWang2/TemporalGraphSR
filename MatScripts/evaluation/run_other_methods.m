% This script is for runing and evaluating the other methods: linear interpolation and spline interpolation 
% Although the interpolation methods do not need the GCN. This script also
% need too be launched after the '\Data\trained_data.mat'
% are already generated by the python scripts. 

% For reviews, the '\Data\trained_data.mat' are also uploaded in the
% attachment, so you can run this scripts without retraining the GCN

% run /init.m before this script to initialize path


clear



% if set load_answers=true, this script will load the pre-calculated answer and
% jump to plotting figures.
load_answers = false;

trained_data = load('..//..//Data//trained_data.mat');
mpc = case33bw_modi;
BBM = cal_BBM(mpc);


[nsample, nbus, nt] = size(trained_data.feature_test.vm);
nbranch = size(BBM,2);

% nsample determines how many tests to perform. There are maximum 1000
% tests stored in the '\Data\trained_data.mat', if you want a quick
% glance, you can set a small value, e.g., nsample = 10
nsample = 10;

if ~load_answers
    %% truthvalue
    truthvalue.plineF = zeros(nsample, nbranch, nt);
    for is = 1:nsample
        for it = 1:nt
            truthvalue.plineF(is, :, it) = BBM\reshape(trained_data.label_test.pi(is, :, it), [], 1);
        end
    end
    truthvalue.vm(1:nsample,:,:) = trained_data.label_test.vm(1:nsample,:,:);

    %% linear interpolations
    vm_interp_linear = zeros(nsample, nbus, nt);
    pi_interp_linear = zeros(nsample, nbus, nt);
    plineF_interp_linear = zeros(nsample, nbranch, nt);
    x_interp = 1:nt;
    for is = 1:nsample
        for ib = 1:nbus
            x = double(trained_data.ava_idx{ib} + 1); %% python index ?0??
            vm = reshape(trained_data.feature_test.vm(is, ib, x), [], 1);
            vm_interp_linear(is, ib, :) = interp1(x,vm, x_interp,'linear','extrap');
            pi = reshape(trained_data.feature_test.pi(is, ib, x), [], 1);
            pi_interp_linear(is, ib, :) = interp1(x,pi,x_interp,'linear','extrap');
        end
    end

    for is = 1:nsample
        for it = 1:nt
            plineF_interp_linear(is, :, it) = BBM\reshape(pi_interp_linear(is, :, it),[],1);
        end
    end


    SR.plineF = plineF_interp_linear;
    SR.vm = vm_interp_linear;
    result_interp_linear = evaluate(SR, truthvalue, nsample);


    %% spline interpolations
    vm_interp_spline = zeros(nsample, nbus, nt);
    pi_interp_spline = zeros(nsample, nbus, nt);
    plineF_interp_spline = zeros(nsample, nbranch, nt);
    x_interp = 1:nt;
    for is = 1:nsample
        for ib = 1:nbus
            x = double(trained_data.ava_idx{ib} + 1); 
            vm = reshape(trained_data.feature_test.vm(is, ib, x), [], 1);
            vm_interp_spline(is, ib, :) = interp1(x,vm, x_interp,'spline');
            pi = reshape(trained_data.feature_test.pi(is, ib, x), [], 1);
            pi_interp_spline(is, ib, :) = interp1(x,pi, x_interp,'spline');    
        end
    end

    for is = 1:nsample
        for it = 1:nt
        plineF_interp_spline(is, :, it) = BBM\reshape(pi_interp_spline(is,:,it), [], 1);
        end
    end

    SR.plineF = plineF_interp_spline;
    SR.vm = vm_interp_spline;
    result_interp_spline = evaluate(SR, truthvalue, nsample);

    save('result_other.mat','result_interp_spline', 'result_interp_linear','vm_interp_spline','plineF_interp_spline','vm_interp_linear','plineF_interp_linear')
else
    load('result_other.mat')
end


figure(2)
subplot(1,3,1)
b = reshape(trained_data.feature_test.vm(1,:,:),33,[]);
imagesc(b)
colorbar
caxis([0.90 1])
xlabel('ʱ�� t')
ylabel('�ڵ���')

subplot(1,3,2)
b = reshape(trained_data.label_test.vm(1,:,:),33,[]);
imagesc(b)
colorbar
caxis([0.90 1])
xlabel('ʱ�� t')
ylabel('�ڵ���')

subplot(1,3,3)
b = reshape(vm_interp_spline(1,:,:),33,[]);
imagesc(b)
colorbar
caxis([0.90 1])
xlabel('ʱ�� t')
ylabel('�ڵ���')


figure(2)
    width=1200;
    height=400;
    left=200;
    bottem=100;
    set(gcf,'position',[left,bottem,width,height])
    subplot(1,3,1)
        b = reshape(vm_interp_linear(1,:,:),33,[]);
        imagesc(b)
        colorbar('southoutside')
        caxis([0.90 1])
        title('(d) Linear interpolation' )
        xlabel('Time')
        ylabel('Node number')
        set(gca, 'xtick', [1,8:8:64])
        set(gca, 'ytick', [1,5:5:30,33])
        set(gca, 'fontname', 'times')
        
        
    subplot(1,3,2)
        b = reshape(vm_interp_spline(1,:,:),33,[]);
        imagesc(b)
        colorbar('southoutside')
        caxis([0.90 1])
        title('(e) Spline interpolation' )
        xlabel('Time')
        ylabel('Node number')
        set(gca, 'xtick', [1,8:8:64])
        set(gca, 'ytick', [1,5:5:30,33])
        set(gca, 'fontname', 'times')

    subplot(1,3,3)
        b = reshape(trained_data.label_test.vm(1,:,:),33,[]);
        imagesc(b)
        colorbar('southoutside')
        caxis([0.90 1])
        title('(c) HTR labels' )
        xlabel('Time')
        ylabel('Node number')
        set(gca, 'xtick', [1,8:8:64])
        set(gca, 'ytick', [1,5:5:30,33])
        set(gca, 'fontname', 'times')

    
        
   


disp('------------Method: Linear Interpolation----------------')
disp(['MSE of voltage magnitude: '  num2str(mean(result_interp_linear.vm_mse))])    
disp(['MSE of active line power flow: ' num2str(mean(result_interp_linear.plineF_mse))])
disp(['MSE overall: ' num2str(mean(result_interp_linear.overall_mse))])   


disp('----------------Method: Spline Interpolation--------------')
disp(['MSE of voltage magnitude: '  num2str(mean(result_interp_spline.vm_mse))])    
disp(['MSE of active line power flow: ' num2str(mean(result_interp_spline.plineF_mse))])
disp(['MSE overall: ' num2str(mean(result_interp_spline.overall_mse))])   

       
        
            
            
        
