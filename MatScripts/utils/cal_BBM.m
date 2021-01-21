function BBM = cal_BBM(mpc)
    nbus = size(mpc.bus,1);
    nbranch = size(mpc.branch,1);
    BBM = zeros(nbus, nbranch);
    for ib = 1:nbranch
        fb = mpc.branch(ib,1);
        tb = mpc.branch(ib,2);
        BBM(fb, ib) = 1;
        BBM(tb, ib) = -1;
    end
    