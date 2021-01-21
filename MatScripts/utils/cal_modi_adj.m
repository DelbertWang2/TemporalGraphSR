%% this script calculate modified adjacent matirx: I + adj

function modi_adj = cal_modi_adj(mpc)
    branch = mpc.branch;
    nbus = size(mpc.bus,1);

    %% cal adj
    adj = zeros(nbus);

    for ib = 1:size(branch,1)
        i = branch(ib,1);
        j = branch(ib,2);
        adj(i,j) = 1;
        adj(j,i) = 1;
    end

    modi_adj = adj + eye(nbus);


