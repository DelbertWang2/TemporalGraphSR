function raw = gen_samples()
    %% this script generates continous measurement data via power flow 
    % notice that this script only performs the simulation and generate the raw data /Data/rawdata.mat. The low
    % temporal resolution features and high temporal resolution labels are
    % constructed in PythonScripts/genSamples/ with the raw data simulated
    % here.





    % bus has the same meanings with 'node' in the paper
    
    bus_load2 = [18:33]; % node 18~33 follows the load shape 2
    bus_load1 = [2:17]; % node 2~17 follows the load shape 1

    bus_solar = [7 8 24 25 30 32]; % node 7 8 24 25 30 32 have solar generations
    p_solar(bus_solar) = [180 250 250 250 150 200];  % in kW, not in MW

    t_step = 300; 
    t_end = 24*3600*(t_step/3)-t_step;
    t = 1:t_step:t_end;

    mpc = case33bw_modi;
    nbus = size(mpc.bus, 1);
    BBM = cal_BBM(mpc);
    
    
    % the idx of columns in MATPOWER format casefile
    bus_map.pd = 3;
    bus_map.qd = 4; 
    bus_map.vm = 8; 
    bus_map.va = 9; 

    gen_map.bus = 1;
    gen_map.pg = 2;
    gen_map.qg = 3; 

    bran_map.plineF = 14;
    bran_map.qlineF = 15;
    bran_map.plineT = 16;
    bran_map.qlineF = 17;



    raw.vm = zeros(nbus, length(t));
    raw.pi = zeros(nbus, length(t));

    for it = 1:length(t)

        mpc = case33bw_modi;
        [load, gen] = get_loadshape(t(it));
        load1 = load(1);
        load2 = load(2);
        solar = gen; % get the basic realtime load ratio and generation ratio
       
        
        sin_l = sin(0.8*it) + sin(2*it);
        sin_h = sin(0.5*it) + sin(2.5*it);
        
        % adjust the standard load in IEEE 33 node system with real time
        % load ratios
        for b = bus_load1
            mpc.bus(b, bus_map.pd) = mpc.bus(b, bus_map.pd) * load1*(1+0.1*sin_l+0.005*randn(1));  
            mpc.bus(b, bus_map.qd) = mpc.bus(b, bus_map.qd) * load1*(1+0.1*sin_l+0.005*randn(1));
        end
        
        for b = bus_load2
            mpc.bus(b, bus_map.pd) = mpc.bus(b, bus_map.pd) * load2*(1+0.2*sin_h+0.005*randn(1));
            mpc.bus(b, bus_map.qd) = mpc.bus(b, bus_map.qd) * load2*(1+0.2*sin_h+0.005*randn(1));
        end
        
        
        igen = 2;
        for b = bus_solar
            mpc.gen(igen, :) =  mpc.gen(1, :);
            mpc.gencost(igen, :) =  mpc.gencost(1, :);
            mpc.gen(igen, gen_map.bus) =  b;
            mpc.gen(igen, gen_map.pg) = p_solar(b)*solar*(1+0.2*sin_h+0.01*randn(1)) /1000; % ?????MW
            igen = igen + 1;
        end

        % run power flow 
        pf = runpf(mpc);

        % get the measurements
        raw.vm(:,it) = pf.bus(:, bus_map.vm);
        pd = pf.bus(:, bus_map.pd);

        pg = zeros(nbus, 1);
        pg(pf.gen(:,gen_map.bus)) = pf.gen(:, gen_map.pg); % pl and pg are p_injections
        % we have to use line injections.

        plineF = pf.branch(:,bran_map.plineF);
        plineT = pf.branch(:,bran_map.plineT);



        raw.pi(:,it) = pg - pd;
        raw.plineF(:,it) = plineF;
        raw.plineT(:,it) = plineT;
        raw.t = t;
        raw.load1(it) = load1;
        raw.load2(it) = load2;
        raw.solar(it) = solar;
    end

    % transform edge features to node features
    raw.plineF_pi = BBM * raw.plineF;
    raw.plineT_pi = BBM * raw.plineT; 





    
    
    