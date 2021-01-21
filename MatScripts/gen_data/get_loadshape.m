%% this script gives the load shape for a specific time t

function [load, gen] = get_loadshape(time)

    dailyload1_base = [0 0.78
        2 0.71
        4 0.72
        6 0.76
        8 0.78
        10 0.98
        12 0.95
        14 1
        16 0.97
        20 0.8
        23 0.86
        24 0.78];
    
    dailyload2_base = [0 0.78
        2 0.71
        4 0.72
        6 0.76
        8 0.78
        10 0.8
        12 0.8
        14 0.7
        16 0.8
        20 0.9
        21 1
        23 0.86
        24 0.78];

    dailysolar_base = [0 0.02
        2 0.02
        3 0.02
        4 0.02
        5 0.1
        6 0.3
        8 0.8
        10 0.8
        12 0.9
        14 1
        16 0.9
        18 0.9
        20 0.3
        21 0.1
        22 0.02
        23 0.02
        24 0.02];

    load_day = [0 0.8
        10 0.8
        20 0.81
        30 0.85
        40 0.8
        50 0.86
        60 0.9
        70 0.8
        80 0.75
        90 0.9
        100 0.9];
    
    day = time/24/3600;
    hour = (time - floor(day)*24*3600)/3600;
    
    
    fac_day = interp1(load_day(:,1), load_day(:,2), day, 'spline');
    load1 = fac_day * interp1(dailyload1_base(:,1), dailyload1_base(:,2), hour, 'spline');
    load2 = fac_day * interp1(dailyload2_base(:,1), dailyload2_base(:,2), hour, 'spline');
    solar = fac_day * interp1(dailysolar_base(:,1), dailysolar_base(:,2), hour, 'spline');
    
    
    load = [load1; load2];
    gen = [solar];
    
end
    
    
    
    
   
    
    
    
    
    
