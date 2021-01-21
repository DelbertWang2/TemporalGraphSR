%% Run this script to generate raw simulation dataset,
% and calculate the modified adjacent matrix and incidence matrix of IEEE 33 node system


mpc = case33bw_modi;
modi_adj = cal_modi_adj(mpc); %modified adjacent matrix
BBM = cal_BBM(mpc); % incidence matrix
raw = gen_samples();

save('..//..//Data//modi_adj.mat','modi_adj');
save('..//..//Data//BBM.mat','BBM');
save('..//..//Data//rawdata.mat','raw');

