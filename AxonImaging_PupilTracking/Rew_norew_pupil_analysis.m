%%Analysis of Pupil area recordings from DAT-Cre mice navigating VR environments for reward
%Chad Heer; Sheffield lab

%1. 1979
%2. 1980
%3. 009
%4. 011
%5. 012
%6. 014

clear all

%% 1979

%load in out put of facemap, split_behavior, and good behavior
pupil_1979.rew_norew = load('1979/pupil data/1979_rew_norew_proc.mat');
pupil_1979.rew_norew.beh = load('1979/Behavior/1979_rew_norew_splitbeh.mat');
pupil_1979.rew_norew.good_beh = load('1979/Behavior/1979_rew_norew_splitbeh_good_behavior.mat');

%designat start of tasks
rew_start = 1;
no_rew_start = 10001;
re_rew_start = 22001;
end_frame = 32000;
bins = 50;

%bin pupil data for each task
tasks = ["rewarded", "unrewarded", "rerewarded"];

[lap_pupil.m1979{1}(:,:),beh.m1979{1}, behavior.m1979.rew]  = bin_pupil(pupil_1979.rew_norew, bins, [rew_start:no_rew_start-1], pupil_1979.rew_norew.good_beh);

[lap_pupil.m1979{2}(:,:),beh.m1979{2}, behavior.m1979.norew]= bin_pupil(pupil_1979.rew_norew, bins, [no_rew_start:re_rew_start-1], pupil_1979.rew_norew.good_beh, nanmean(beh.m1979{1}.rew_location));

[lap_pupil.m1979{3}(:,:),beh.m1979{3}, behavior.m1979.rerew] = bin_pupil(pupil_1979.rew_norew, bins, [re_rew_start:end_frame], pupil_1979.rew_norew.good_beh);

mean_lap_pupil = plot_pupil_means(lap_pupil.m1979, tasks);

title('1979 pupil')

%% 1979 saline day

pupil_1979.saline = load('1979/pupil data/1979_saline_proc.mat');
pupil_1979.saline.beh = load('1979/Behavior/1979_saline_splitbeh.mat');
pupil_1979.saline.good_beh = load('1979/Behavior/1979_saline_goodbeh.mat');


beforesaline_start = 1;
aftersaline_start = 12001;
end_frame = 32000;
bins = 50;


tasks = ["rewarded", "unrewarded", "rerewarded", "before saline CNO", "after saline CNO"];

[lap_pupil.m1979{4}(:,:),beh.m1979{4}, behavior.m1979.bef_saline]  = bin_pupil(pupil_1979.saline, bins, [beforesaline_start:aftersaline_start-1], pupil_1979.saline.good_beh);

[lap_pupil.m1979{5}(:,:),beh.m1979{5}, behavior.m1979.aft_saline]= bin_pupil(pupil_1979.saline, bins, [aftersaline_start:end_frame], pupil_1979.saline.good_beh);

mean_lap_pupil = plot_pupil_means(lap_pupil.m1979, tasks);

title('1979 saline pupil')


%% 1979 5mg CNO day

pupil_1979.CNO_5mg = load('1979/pupil data/1979_5mgCNO_proc.mat');
pupil_1979.CNO_5mg.beh = load('1979/Behavior/1979_5mgCNO_splitbeh.mat');
pupil_1979.CNO_5mg.good_beh = load('1979/Behavior/1979_5mgCNO_goodbeh.mat');

beforeCNO_start = 1;
afterCNO_start = 12001;
end_frame = 36000
bins = 50;


tasks = ["rewarded", "unrewarded", "rerewarded",  "before saline CNO", "after saline CNO","before 5mg/kg CNO", "after 5mg/kg CNO"];

[lap_pupil.m1979{6}(:,:),beh.m1979{6}, behavior.m1979.bef_CNO]  = bin_pupil(pupil_1979.CNO_5mg, bins, [beforeCNO_start:afterCNO_start-1], pupil_1979.CNO_5mg.good_beh);

[lap_pupil.m1979{7}(:,:),beh.m1979{7}, behavior.m1979.aft_CNO]= bin_pupil(pupil_1979.CNO_5mg, bins, [afterCNO_start:end_frame], pupil_1979.CNO_5mg.good_beh);

mean_lap_pupil = plot_pupil_means(lap_pupil.m1979, tasks);

title('1979 CNO pupil')


%% 1980

pupil_1980.rew_norew = load('1980/pupil data/1980_rew_norew_proc.mat');
pupil_1980.rew_norew.beh = load('1980/Behavior/1980_rew_norew_splitbeh.mat');
pupil_1980.rew_norew.good_beh = load('1980/Behavior/1980_rew_norew_splitbeh_good_behavior.mat');

rew_start = 1;
no_rew_start = 10001;
re_rew_start = 22001;
end_frame = 34000;
bins = 50;

tasks = ["rewarded", "unrewarded", "rerewarded"];

[lap_pupil.m1980{1}(:,:),beh.m1980{1}, behavior.m1980.rew] = bin_pupil(pupil_1980.rew_norew, bins, [rew_start:no_rew_start-1], pupil_1980.rew_norew.good_beh);

[lap_pupil.m1980{2}(:,:),beh.m1980{2}, behavior.m1980.norew]= bin_pupil(pupil_1980.rew_norew, bins, [no_rew_start:re_rew_start-1], pupil_1980.rew_norew.good_beh);

[lap_pupil.m1980{3}(:,:),beh.m1980{3}, behavior.m1980.rerew] = bin_pupil(pupil_1980.rew_norew, bins, [re_rew_start:end_frame], pupil_1980.rew_norew.good_beh);

mean_lap_pupil = plot_pupil_means(lap_pupil.m1980, tasks);

title('1980 pupil')



%% 1980 saline day

pupil_1980.saline = load('1980/pupil data/1980_saline_proc.mat');
pupil_1980.saline.beh = load('1980/Behavior/1980_saline_splitbeh.mat');
pupil_1980.saline.good_beh = load('1980/Behavior/1980_saline_goodbeh.mat');

beforesaline_start = 1;
aftersaline_start = 12001;
end_frame = 32000
bins = 50;


tasks = ["rewarded", "unrewarded", "rerewarded", "before saline CNO", "after saline CNO"];

[lap_pupil.m1980{4}(:,:),beh.m1980{4}, behavior.m1980.bef_saline]  = bin_pupil(pupil_1980.saline, bins, [beforesaline_start:aftersaline_start-1], pupil_1980.saline.good_beh);

[lap_pupil.m1980{5}(:,:),beh.m1980{5}, behavior.m1980.aft_saline]= bin_pupil(pupil_1980.saline, bins, [aftersaline_start:end_frame], pupil_1980.saline.good_beh);

mean_lap_pupil = plot_pupil_means(lap_pupil.m1980, tasks);

title('1980 saline pupil')



%% 1980 5mg CNO day

pupil_1980.CNO_5mg = load('1980/pupil data/1980_5mgCNO_proc.mat');
pupil_1980.CNO_5mg.beh = load('1980/Behavior/1980_5mgCNO_splitbeh.mat');
pupil_1980.CNO_5mg.good_beh = load('1980/Behavior/1980_5mgCNO_goodbeh.mat');

beforeCNO_start = 1;
afterCNO_start = 12001;
end_frame = 60000
bins = 50;


tasks = ["rewarded", "unrewarded", "rerewarded",  "before saline CNO", "after saline CNO","before 5mg/kg CNO", "after 5mg/kg CNO"];

[lap_pupil.m1980{6}(:,:),beh.m1980{6}, behavior.m1980.bef_CNO]  = bin_pupil(pupil_1980.CNO_5mg, bins, [beforeCNO_start:afterCNO_start-1], pupil_1980.CNO_5mg.good_beh);

[lap_pupil.m1980{7}(:,:),beh.m1980{7}, behavior.m1980.aft_CNO]= bin_pupil(pupil_1980.CNO_5mg, bins, [afterCNO_start:end_frame], pupil_1980.CNO_5mg.good_beh);

mean_lap_pupil = plot_pupil_means(lap_pupil.m1980, tasks);

title('1980 CNO pupil')


%% 009

pupil_009.rew_norew = load('009/pupil data/009_rew_norew_proc.mat');
pupil_009.rew_norew.beh = load('009/Behavior/009_rew_norew_splitbeh.mat');
pupil_009.rew_norew.good_beh = load('009/Behavior/009_rew_norew_splitbeh_good_behavior.mat');

rew_start = 1;
no_rew_start = 12001;
re_rew_start = 27001;
end_frame = 39000;
bins = 50;

tasks = ["rewarded", "unrewarded", "rerewarded"];

[lap_pupil.m009{1}(:,:),beh.m009{1}, behavior.m009.rew] = bin_pupil(pupil_009.rew_norew, bins, [rew_start:no_rew_start-1], pupil_009.rew_norew.good_beh);

[lap_pupil.m009{2}(:,:),beh.m009{2}, behavior.m009.norew]= bin_pupil(pupil_009.rew_norew, bins, [no_rew_start:re_rew_start-1], pupil_009.rew_norew.good_beh);

[lap_pupil.m009{3}(:,:),beh.m009{3}, behavior.m009.rerew] = bin_pupil(pupil_009.rew_norew, bins, [re_rew_start:end_frame], pupil_009.rew_norew.good_beh);

mean_lap_pupil = plot_pupil_means(lap_pupil.m009, tasks);

title('009 pupil')


%% 009 saline day

pupil_009.saline = load('009/pupil data/009_saline_proc.mat');
pupil_009.saline.beh = load('009/Behavior/009_saline_splitbeh.mat');
pupil_009.saline.good_beh = load('009/Behavior/009_saline_goodbeh.mat');

beforesaline_start = 1;
aftersaline_start = 12001;
end_frame = 32000
bins = 50;


tasks = ["rewarded", "unrewarded", "rerewarded", "before saline CNO", "after saline CNO"];

[lap_pupil.m009{4}(:,:),beh.m009{4}, behavior.m009.bef_saline]  = bin_pupil(pupil_009.saline, bins, [beforesaline_start:aftersaline_start-1], pupil_009.saline.good_beh);

[lap_pupil.m009{5}(:,:),beh.m009{5}, behavior.m009.aft_saline]= bin_pupil(pupil_009.saline, bins, [aftersaline_start:end_frame], pupil_009.saline.good_beh);

mean_lap_pupil = plot_pupil_means(lap_pupil.m009, tasks);

title('009 saline pupil')



%% 009 5mg CNO day

pupil_009.CNO_5mg = load('009/pupil data/009_5mgCNO_proc.mat');
pupil_009.CNO_5mg.beh = load('009/Behavior/009_5mgCNO_splitbeh.mat');
pupil_009.CNO_5mg.good_beh = load('009/Behavior/009_5mgCNO_goodbeh.mat');

beforeCNO_start = 1;
afterCNO_start = 12001;
end_frame = 32000;
bins = 50;


tasks = ["rewarded", "unrewarded", "rerewarded",  "before saline CNO", "after saline CNO","before 5mg/kg CNO", "after 5mg/kg CNO"];

[lap_pupil.m009{6}(:,:),beh.m009{6}, behavior.m009.bef_CNO]  = bin_pupil(pupil_009.CNO_5mg, bins, [beforeCNO_start:afterCNO_start-1], pupil_009.CNO_5mg.good_beh);

[lap_pupil.m009{7}(:,:),beh.m009{7}, behavior.m009.aft_CNO]= bin_pupil(pupil_009.CNO_5mg, bins, [afterCNO_start:end_frame], pupil_009.CNO_5mg.good_beh);

mean_lap_pupil = plot_pupil_means(lap_pupil.m009, tasks);

title('009 CNO pupil')

%% 011
pupil_011.rew_norew = load('011/pupil data/011_rew_norew_proc.mat');
pupil_011.rew_norew.beh = load('011/Behavior/011_rew_norew_splitbeh.mat');
pupil_011.rew_norew.good_beh = load('011/Behavior/011_rew_norew_splitbeh_good_behavior.mat');

rew_start = 1;
no_rew_start = 12001;
re_rew_start = 27001;
end_frame = 39000;
bins = 50;

tasks = ["rewarded", "unrewarded", "rerewarded"];

[lap_pupil.m011{1}(:,:),beh.m011{1}, behavior.m011.rew] = bin_pupil(pupil_011.rew_norew, bins, [rew_start:no_rew_start-1], pupil_011.rew_norew.good_beh);

[lap_pupil.m011{2}(:,:),beh.m011{2}, behavior.m011.norew]= bin_pupil(pupil_011.rew_norew, bins, [no_rew_start:re_rew_start-1], pupil_011.rew_norew.good_beh);

[lap_pupil.m011{3}(:,:),beh.m011{3}, behavior.m011.rerew] = bin_pupil(pupil_011.rew_norew, bins, [re_rew_start:end_frame], pupil_011.rew_norew.good_beh);

mean_lap_pupil = plot_pupil_means(lap_pupil.m011, tasks);

title('011 pupil')


%% 011 saline day

pupil_011.saline = load('011/pupil data/011_saline_proc.mat');
pupil_011.saline.beh = load('011/Behavior/011_saline_splitbeh.mat');
pupil_011.saline.good_beh = load('011/Behavior/011_saline_goodbeh.mat');

beforesaline_start = 1;
aftersaline_start = 12001;
end_frame = 32000
bins = 50;


tasks = ["rewarded", "unrewarded", "rerewarded", "before saline CNO", "after saline CNO"];

[lap_pupil.m011{4}(:,:),beh.m011{4}, behavior.m011.bef_saline]  = bin_pupil(pupil_011.saline, bins, [beforesaline_start:aftersaline_start-1], pupil_011.saline.good_beh);

[lap_pupil.m011{5}(:,:),beh.m011{5}, behavior.m011.aft_saline]= bin_pupil(pupil_011.saline, bins, [aftersaline_start:end_frame], pupil_011.saline.good_beh);

mean_lap_pupil = plot_pupil_means(lap_pupil.m011, tasks);

title('011 saline pupil')


%% 011 5mg CNO day

pupil_011.CNO_5mg = load('011/pupil data/011_5mgCNO_proc.mat');
pupil_011.CNO_5mg.beh = load('011/Behavior/011_5mgCNO_splitbeh.mat');
pupil_011.CNO_5mg.good_beh = load('011/Behavior/011_5mgCNO_goodbeh.mat');

beforeCNO_start = 1;
afterCNO_start = 12001;
end_frame = 32000;
bins = 50;


tasks = ["rewarded", "unrewarded", "rerewarded",  "before saline CNO", "after saline CNO","before 5mg/kg CNO", "after 5mg/kg CNO"];

[lap_pupil.m011{6}(:,:),beh.m011{6}, behavior.m011.bef_CNO]  = bin_pupil(pupil_011.CNO_5mg, bins, [beforeCNO_start:afterCNO_start-1], pupil_011.CNO_5mg.good_beh);

[lap_pupil.m011{7}(:,:),beh.m011{7}, behavior.m011.aft_CNO]= bin_pupil(pupil_011.CNO_5mg, bins, [afterCNO_start:end_frame], pupil_011.CNO_5mg.good_beh);

mean_lap_pupil = plot_pupil_means(lap_pupil.m011, tasks);

title('011 CNO pupil')


%% 012

pupil_012.rew_norew = load('012/pupil data/012_rew_norew_proc.mat');
pupil_012.rew_norew.beh = load('012/Behavior/012_rew_norew_splitbeh.mat');
pupil_012.rew_norew.good_beh = load('012/Behavior/012_rew_norew_splitbeh_good_behavior.mat');

rew_start = 1;
no_rew_start = 12001;
re_rew_start = 27001;
end_frame = 39000;
bins = 50;

tasks = ["rewarded", "unrewarded", "rerewarded"];

[lap_pupil.m012{1}(:,:),beh.m012{1}, behavior.m012.rew] = bin_pupil(pupil_012.rew_norew, bins, [rew_start:no_rew_start-1], pupil_012.rew_norew.good_beh);

[lap_pupil.m012{2}(:,:),beh.m012{2}, behavior.m012.norew]= bin_pupil(pupil_012.rew_norew, bins, [no_rew_start:re_rew_start-1], pupil_012.rew_norew.good_beh);

[lap_pupil.m012{3}(:,:),beh.m012{3}, behavior.m012.rerew] = bin_pupil(pupil_012.rew_norew, bins, [re_rew_start:end_frame], pupil_012.rew_norew.good_beh);

mean_lap_pupil = plot_pupil_means(lap_pupil.m012, tasks);

% 
%% 012 saline day

pupil_012.saline = load('012/pupil data/012_saline_proc.mat');
pupil_012.saline.beh = load('012/Behavior/012_saline_splitbeh.mat');
pupil_012.saline.good_beh = load('012/Behavior/012_saline_goodbeh.mat');

beforesaline_start = 1;
aftersaline_start = 12001;
end_frame = 32000
bins = 50;


tasks = ["rewarded", "unrewarded", "rerewarded", "before saline CNO", "after saline CNO"];

[lap_pupil.m012{4}(:,:),beh.m012{4}, behavior.m012.bef_saline]  = bin_pupil(pupil_012.saline, bins, [beforesaline_start:aftersaline_start-1], pupil_012.saline.good_beh);

[lap_pupil.m012{5}(:,:),beh.m012{5}, behavior.m012.aft_saline]= bin_pupil(pupil_012.saline, bins, [aftersaline_start:end_frame], pupil_012.saline.good_beh);

mean_lap_pupil = plot_pupil_means(lap_pupil.m012, tasks);

title('012 saline pupil')


%% 012 5mg CNO day

pupil_012.CNO_5mg = load('012/pupil data/012_5mgCNO_proc.mat');
pupil_012.CNO_5mg.beh = load('012/Behavior/012_5mgCNO_splitbeh.mat');
pupil_012.CNO_5mg.good_beh = load('012/Behavior/012_5mgCNO_goodbeh.mat');

beforeCNO_start = 1;
afterCNO_start = 12001;
end_frame = 32000;
bins = 50;


tasks = ["rewarded", "unrewarded", "rerewarded",  "before saline CNO", "after saline CNO","before 5mg/kg CNO", "after 5mg/kg CNO"];

[lap_pupil.m012{6}(:,:),beh.m012{6}, behavior.m012.bef_CNO]  = bin_pupil(pupil_012.CNO_5mg, bins, [beforeCNO_start:afterCNO_start-1], pupil_012.CNO_5mg.good_beh);

[lap_pupil.m012{7}(:,:),beh.m012{7}, behavior.m012.aft_CNO]= bin_pupil(pupil_012.CNO_5mg, bins, [afterCNO_start:end_frame], pupil_012.CNO_5mg.good_beh);

mean_lap_pupil = plot_pupil_means(lap_pupil.m012, tasks);

title('012 CNO pupil')

%% 014

pupil_014.rew_norew = load('014/pupil data/014_rew_norew_proc.mat');
pupil_014.rew_norew.beh = load('014/Behavior/014_rew_norew_splitbeh.mat');
pupil_014.rew_norew.good_beh = load('014/Behavior/014_rew_norew_splitbeh_good_behavior.mat');

rew_start = 1;
no_rew_start = 12001;
re_rew_start = 27001;
end_frame = 39000;
bins = 50;

tasks = ["rewarded", "unrewarded", "rerewarded"];

[lap_pupil.m014{1}(:,:),beh.m014{1}, behavior.m014.rew] = bin_pupil(pupil_014.rew_norew, bins, [rew_start:no_rew_start-1], pupil_014.rew_norew.good_beh);

[lap_pupil.m014{2}(:,:),beh.m014{2}, behavior.m014.norew]= bin_pupil(pupil_014.rew_norew, bins, [no_rew_start:re_rew_start-1], pupil_014.rew_norew.good_beh);

[lap_pupil.m014{3}(:,:),beh.m014{3}, behavior.m014.rerew] = bin_pupil(pupil_014.rew_norew, bins, [re_rew_start:end_frame], pupil_014.rew_norew.good_beh);

mean_lap_pupil = plot_pupil_means(lap_pupil.m014, tasks);


%% 014 saline day

pupil_014.saline = load('014/pupil data/014_saline_proc.mat');
pupil_014.saline.beh = load('014/Behavior/014_saline_splitbeh.mat');
pupil_014.saline.good_beh = load('014/Behavior/014_saline_goodbeh.mat');

beforesaline_start = 1;
aftersaline_start = 12001;
end_frame = 32000
bins = 50;


tasks = ["rewarded", "unrewarded", "rerewarded", "before saline CNO", "after saline CNO"];

[lap_pupil.m014{4}(:,:),beh.m014{4}, behavior.m014.bef_saline]  = bin_pupil(pupil_014.saline, bins, [beforesaline_start:aftersaline_start-1], pupil_014.saline.good_beh);

[lap_pupil.m014{5}(:,:),beh.m014{5}, behavior.m014.aft_saline]= bin_pupil(pupil_014.saline, bins, [aftersaline_start:end_frame], pupil_014.saline.good_beh);

mean_lap_pupil = plot_pupil_means(lap_pupil.m014, tasks);

title('014 saline pupil')


%% 014 5mg CNO day

pupil_014.CNO_5mg = load('014/pupil data/014_5mgCNO_proc.mat');
pupil_014.CNO_5mg.beh = load('014/Behavior/014_5mgCNO_splitbeh.mat');
pupil_014.CNO_5mg.good_beh = load('014/Behavior/014_5mgCNO_goodbeh.mat');

beforeCNO_start = 1;
afterCNO_start = 12001;
end_frame = 32000;
bins = 50;


tasks = ["rewarded", "unrewarded", "rerewarded",  "before saline CNO", "after saline CNO","before 5mg/kg CNO", "after 5mg/kg CNO"];

[lap_pupil.m014{6}(:,:),beh.m014{6}, behavior.m014.bef_CNO]  = bin_pupil(pupil_014.CNO_5mg, bins, [beforeCNO_start:afterCNO_start-1], pupil_014.CNO_5mg.good_beh);

[lap_pupil.m014{7}(:,:),beh.m014{7}, behavior.m014.aft_CNO]= bin_pupil(pupil_014.CNO_5mg, bins, [afterCNO_start:end_frame], pupil_014.CNO_5mg.good_beh);

mean_lap_pupil = plot_pupil_means(lap_pupil.m014, tasks);

title('014 CNO pupil')


%% 7. plotting normalized lap pupil data 
%initialize variables
norm_lap_pupil = {[],[],[],[],[],[],[]};
unrew_pupil = {[],[]};
licking = norm_lap_pupil;
freezing_ratio = norm_lap_pupil;
mean_velocity = norm_lap_pupil;
com_mean_dist = norm_lap_pupil; 
blink_ratio = norm_lap_pupil; 
mouse_id = norm_lap_pupil;

tasks = ["Rewarded laps", "Unrewarded laps", "rerewarded",  "before saline CNO", "after saline CNO","before 5mg/kg CNO", "after 5mg/kg CNO"];
fn = fieldnames(lap_pupil);


for mouse = 1: length(fn)
    %combine behavioral data from each mouse 
    for task = 1: length(tasks)
        norm_lap_pupil{1,task} = [norm_lap_pupil{1,task}; (lap_pupil.(fn{mouse}){task}./max(lap_pupil.(fn{mouse}){task},[],2))];
        licking{task} = [licking{task}, beh.(fn{mouse}){1,task}.licking];
        freezing_ratio{task} = [freezing_ratio{task}, beh.(fn{mouse}){1,task}.freezing_ratio];
        com_mean_dist{task} = [com_mean_dist{task}, beh.(fn{mouse}){1,task}.com_mean_dist'];
        mean_velocity{task} = [mean_velocity{task}, beh.(fn{mouse}){1,task}.mean_velocity/max(beh.(fn{mouse}){1,task}.mean_velocity)];
        blink_ratio{task} = [blink_ratio{task}, beh.(fn{mouse}){1,task}.blinking_ratio];
        id = strings(1,length(beh.(fn{mouse}){1,task}.mean_velocity));
        id(:) = fn{mouse};
        mouse_id{task} = [mouse_id{task}, id];
    end
    %split unrewarded laps into licking and not licking laps
    unrew_pupil{1} = [unrew_pupil{1}; (lap_pupil.(fn{mouse}){2}(find(beh.(fn{mouse}){1,2}.licking == 1), :)./max(lap_pupil.(fn{mouse}){2}(find(beh.(fn{mouse}){1,2}.licking == 1), :),[],2))];
    unrew_pupil{2} = [unrew_pupil{2}; (lap_pupil.(fn{mouse}){2}(find(beh.(fn{mouse}){1,2}.licking == 0), :)./max(lap_pupil.(fn{mouse}){2}(find(beh.(fn{mouse}){1,2}.licking == 0), :), [],2))];
    
    template.(fn{mouse}) = plot_pupil_means(lap_pupil.(fn{mouse}),tasks);
end

position = 1:5:200;

unrew = ["unrew licks", "unrew nolicks"];

template_mean_pupil = plot_pupil_means(norm_lap_pupil, tasks);

plot_laps_with_mean(position, norm_lap_pupil,tasks, [0.8, 1.0]);

plot_laps_with_mean(position, unrew_pupil,unrew, [0.8, 1]);

plot_pupil_means(unrew_pupil, unrew);

%% 8. plot pupil correlation vs behavioral variables
lap_pupil_corr = {[],[],[],[],[],[],[]};
high_corr_laps = lap_pupil_corr;
low_corr_laps = lap_pupil_corr;
high_corr_licking = lap_pupil_corr;
low_corr_licking = lap_pupil_corr;

for task=1: length(tasks);
    for mouse = 1: length(fn)
        mouse_lap_pupil_corr = {[],[],[],[],[],[],[]};
        %find the correlation behtween each laps bin pupil area with the
        %mean pupil area of all laps in the familiar environment
        for lap = 1:size(lap_pupil.(fn{mouse}){task},1);
            correlation = corrcoef(template.(fn{mouse}){task}, lap_pupil.(fn{mouse}){task}(lap,:),'rows','pairwise');
            mouse_lap_pupil_corr{task} = [mouse_lap_pupil_corr{task}; correlation(1,2)];
            lap_pupil_corr{task} = [lap_pupil_corr{task}; correlation(1,2)];
            
            if lap < size(lap_pupil.(fn{mouse}){task},1) - 2 & sum(beh.(fn{mouse}){task}.licking(lap:lap+2)) == 0
                beh.(fn{mouse}){task}.licking(lap:end) = 0; 
            elseif lap < size(lap_pupil.(fn{mouse}){task},1) - 2
                beh.(fn{mouse}){task}.licking(lap) = 1; 
            end
        end
        %divide laps into highly  and lowly correlated pupil area laps
        corr_thresh(mouse,task) = nanmean(mouse_lap_pupil_corr{task}(:)) - 1.5 * nanstd(mouse_lap_pupil_corr{task}(:));

        high_corr_laps{task} = [high_corr_laps{task}; lap_pupil.(fn{mouse}){task}(find(mouse_lap_pupil_corr{task}(:) >= corr_thresh(mouse,1)),:)./max(lap_pupil.(fn{mouse}){task}(find(mouse_lap_pupil_corr{task}(:) >= corr_thresh(mouse,1)),:),[],2)];
        low_corr_laps{task} = [low_corr_laps{task}; lap_pupil.(fn{mouse}){task}(find(mouse_lap_pupil_corr{task}(:) < corr_thresh(mouse,1)),:)./max(lap_pupil.(fn{mouse}){task}(find(mouse_lap_pupil_corr{task}(:) < corr_thresh(mouse,1)),:),[],2)];

        high_corr_licking{task} = [high_corr_licking{task}, beh.(fn{mouse}){task}.licking(find(mouse_lap_pupil_corr{task}(:) >= corr_thresh(mouse,1)))];
        low_corr_licking{task} = [low_corr_licking{task}, beh.(fn{mouse}){task}.licking(find(mouse_lap_pupil_corr{task}(:) < corr_thresh(mouse,1)))];
    end
    
    
    

    
    figure;
    histogram(lap_pupil_corr{task}(:),20);
    title(tasks{task})

    
    %Plot pupil correlation vs behavioral variables and determine any
    %correlations. 
    figure;
    title(tasks{task})
    subplot(2,2,1)
    scatter(lap_pupil_corr{task}(:),mean_velocity{task})
    lm = fitlm(lap_pupil_corr{task}(:), mean_velocity{task});
    x = anova(lm,'summary');
    hold on
    plot(lap_pupil_corr{task}(:), lm.Fitted)
    title(['r2 = ' num2str(lm.Rsquared.Ordinary) ', p = ' num2str(table2array(x(2,5)))]);
    xlabel('pupil corr')
    ylabel('norm mean velocity')
    
    subplot(2,2,2)
    scatter(lap_pupil_corr{task}(:),freezing_ratio{task})
    lm = fitlm(lap_pupil_corr{task}(:), freezing_ratio{task});
    x = anova(lm,'summary');
    hold on
    plot(lap_pupil_corr{task}(:), lm.Fitted)
    title(['r2 = ' num2str(lm.Rsquared.Ordinary) ', p = ' num2str(table2array(x(2,5)))])
    xlabel('pupil corr')
    ylabel('freezing')
    
    subplot(2,2,3)
    scatter(lap_pupil_corr{task}(:),blink_ratio{task})
    lm = fitlm(lap_pupil_corr{task}(:), blink_ratio{task});
    x = anova(lm,'summary');
    hold on
    plot(lap_pupil_corr{task}(:), lm.Fitted)
    title(['r2 = ' num2str(lm.Rsquared.Ordinary) ', p = ' num2str(table2array(x(2,5)))])   
    xlabel('pupil corr')
    ylabel('blinking')
        
    subplot(2,2,4)
    scatter(lap_pupil_corr{task}(:),com_mean_dist{task})
    lm = fitlm(lap_pupil_corr{task}(:), com_mean_dist{task});
    x = anova(lm,'summary');
    hold on
    plot(lap_pupil_corr{task}(:), lm.Fitted)
    title(['r2 = ' num2str(lm.Rsquared.Ordinary) ', p = ' num2str(table2array(x(2,5)))])  
    xlabel('pupil corr')
    ylabel('pupil movement')
end

%% 9. plot pupil corr vs behavior for unrew no licking laps
%divide unrewarded task into unrewarded licking, unrewarded no licking but
%high pupil corr, and unrewarded no licking and low pupil corr
unrew = ["Unrewarded laps with licks", "Unrewarded lips without licks: high correlation" , "Unrewarded laps without licks: low correlation"]
unrew_pupil{1} = (high_corr_laps{1,2}(find(high_corr_licking{2} == 1), :));
unrew_pupil{2} = (high_corr_laps{1,2}(find(high_corr_licking{2} == 0), :));
unrew_pupil{3} = (low_corr_laps{1,2}(find(low_corr_licking{2} == 0), :));


unrew_nolick.pupil_corr = lap_pupil_corr{2}(find(licking{2} == 1));
unrew_nolick.com_mean_dist = com_mean_dist{2}(find(licking{2} == 1));
unrew_nolick.mean_velocity = mean_velocity{2}(find(licking{2} == 1));
unrew_nolick.blink_ratio = blink_ratio{2}(find(licking{2} == 1));
unrew_nolick.freeze_ratio = freezing_ratio{2}(find(licking{2} == 1));

x = fieldnames(unrew_nolick)
%plot pupil correlation vs behavioral variables for unrewarded/no licking
%laps
figure;
for y = 2:length(x)

    subplot(2,2,y-1)
    scatter(unrew_nolick.pupil_corr, unrew_nolick.(x{y}))
    lm = fitlm(unrew_nolick.pupil_corr, unrew_nolick.(x{y}));
    table = anova(lm,'summary');
    hold on
    plot(unrew_nolick.pupil_corr, lm.Fitted)
    title(['r2 = ' num2str(lm.Rsquared.Ordinary) ', p = ' num2str(table2array(table(2,5)))]);
    xlabel('pupil corr')
    ylabel(x{y})
end

    
plot_laps_with_mean(position, unrew_pupil, unrew, [0.8, 1.0]);


tasks = fieldnames(behavior.m1979);
  
save('for_seetha','behavior')
save('corr_thresh','corr_thresh')

