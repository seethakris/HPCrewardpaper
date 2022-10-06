%% VTA ramping analysis
%Chad Heer; Sheffield Lab 

%Analyis of VTA axon fluorescent activity in mice as they navigate virtual
%environments for rewards. 

%generates the figures and statistics use in Fig 5 and Supplimentary Figs
%19, 20, and 21 

%Sections
%1. Mouse 113 
%2. Mouse 115
%3. Mouse 265
%4. Mouse 223
%5. Mouse 253
%6. Mouse 255
%7. plot means and 95% CI for sectioned data
%8. Plot mean time and position binned activity of all mice 
%9. Align binned data with removal of reward



%% 1. 113 analysis

clear all
close all

%load in behavior and by hand ROI data
load('113/reward_noreward/113_splitbeh.mat');
load('113/reward_noreward/plane2_ROIs_ROIs.mat');
load('113/reward_noreward/113_good_behavior.mat');

%designate imaging plane to use and track length
plane = 2;
tracklength = 200;

%label tasks and frames for each task
tasks = ["rew","no rew-licking","no rew-no lick","re rew", "nov"];
frames{1} = [19:3314]; %rewarded frames
frames{2} = [3425:4745]; %RE high framse
frames{3} = [4748:8218];  %RE low frames  
frames{4} = [8483:12290];  %RR frames
frames{5} = [12500:17333];  %Novel Frames

%grab behavioral variable names
fields = fieldnames(behplane{1,plane});

%initialize variables
combined_bin_F = {[],[],[],[],[]};
combined_Tbin_F = combined_bin_F
combined_bin_F_sub_baseline = combined_bin_F;
combined_Tbin_F_sub_baseline = combined_bin_F;
early_late_Tbin_F = combined_bin_F;
combined_max_bin_F_sub_base = [];
combined_max_Tbin_F_sub_base = [];
combined_lap_max_F = [];

all_tasks = [];

%divide track into start, mid, and end 
sections = {1:10;11:30;31:40};
section_means = zeros(7,5,3);
section_Tmeans = zeros(7,5,3);
mouse = 1;

%For the best ROI
for roi=5: size(data.F,2)
    for task = 1: length(tasks)                                            
        %for each task, grab the behavior for that task for the imaging plane
        for field = 1: length(fields)-1
            behavior{task}.(fields{field}) = behplane{1,plane}.(fields{field})(frames{task});
        end

        %bin the fluorescence data into positional and time bins for laps
        [bin_mean_F{task},Tbin_mean_F{task}, lap_max_F{task},AUC{task}, time2reward{1,task}, postrewardtime{1,task}, num_licks{1,task}] = binFbyPosition(smooth(data.F(frames{task},roi),7,'sgolay',5), behavior{task}, tracklength, frames{task});
        
        %find max value for each lap
        [max_bin_F{task}(:,1),max_bin_F{task}(:,2)] = max(bin_mean_F{task},[],2);
        [max_Tbin_F{task}(:,1),max_Tbin_F{task}(:,2)] = max(Tbin_mean_F{task},[],2);
        
        %subtract the baseline fluorescence for each lap
        baseline_sub_binned_F{task} = bin_mean_F{task} - min(bin_mean_F{task}(:,:),[],2);
        baseline_sub_Tbinned_F{task} = Tbin_mean_F{task} - min(Tbin_mean_F{task}(:,:),[],2);
        
        %for each lap find the area under the curve of the ramping activity
        for lap = 1: size(bin_mean_F{task})
            position = 1:5:tracklength;
            position = position(~isnan(baseline_sub_binned_F{task}(lap,:))); 
            base_sub_bin_F_nanless = baseline_sub_binned_F{task}(lap,~isnan(baseline_sub_binned_F{task}(lap,:)));

            position = 1:5:tracklength;
            position = position(~isnan(baseline_sub_Tbinned_F{task}(lap,:))); 
            base_sub_Tbin_F_nanless = baseline_sub_Tbinned_F{task}(lap,~isnan(baseline_sub_Tbinned_F{task}(lap,:)));
        end
        
    end
    
    %find and plot the mean place and time binned activity
    mean_bin_F = plot_pupil_means(bin_mean_F, tasks);
    mean_Tbin_F = plot_pupil_means(Tbin_mean_F, tasks);
    
    mean_bin_F_sub_baseline = plot_pupil_means(baseline_sub_binned_F, tasks);
    mean_Tbin_F_sub_baseline = plot_pupil_means(baseline_sub_Tbinned_F, tasks);

    
    for task = 1:length(tasks)
        all_tasks = [all_tasks; baseline_sub_Tbinned_F{task}];
        
        %normalize and combine data
        combined_bin_F{task} = [combined_bin_F{task};(bin_mean_F{task}/max(cell2mat(mean_bin_F)))];
        combined_Tbin_F{task} = [combined_Tbin_F{task};(Tbin_mean_F{task}/max(cell2mat(mean_Tbin_F)))];
        combined_bin_F_sub_baseline{task} = [combined_bin_F_sub_baseline{task};(baseline_sub_binned_F{task}/max(cell2mat(mean_bin_F_sub_baseline)))];
        combined_Tbin_F_sub_baseline{task} = [combined_Tbin_F_sub_baseline{task};(baseline_sub_Tbinned_F{task}/max(cell2mat(mean_Tbin_F_sub_baseline)))];
            
        %find the max of each lap and normalize and combine data
        max_Tbin_F_sub_base_raw{1,task} = max(baseline_sub_Tbinned_F{task},[],2);
        max_Tbin_F_sub_base{1,task}(:,1) = max_Tbin_F_sub_base_raw{1,task}/nanmean(max_Tbin_F_sub_base_raw{1,1}); 
        [placeholder, max_Tbin_F_sub_base{1,task}(:,2)] = max(baseline_sub_Tbinned_F{task},[],2);
        max_Tbin_F_sub_base{1,task}(:,3) = max_Tbin_F_sub_base{1,task}(:,1).*max_Tbin_F_sub_base{1,task}(:,2);
        
        combined_max_Tbin_F_sub_base = [combined_max_Tbin_F_sub_base; max_Tbin_F_sub_base{1,task}];
        combined_lap_max_F = [combined_lap_max_F; lap_max_F{task}'];

         
        %find the mean fluorescence at start, mid, and end of track 
        for section=1: size(sections,1)
            section_means(mouse, task, section) = nanmean(nanmean((baseline_sub_binned_F{task}(:,sections{section})/max(cell2mat(mean_bin_F_sub_baseline)))));
            section_Tmeans(mouse, task, section) = nanmean(nanmean((baseline_sub_Tbinned_F{task}(:,sections{section})/max(cell2mat(mean_Tbin_F_sub_baseline)))));
        end
    end
    
    % find slope of ramping activity
    [lap_slope{1},line_r2{1},exp_coeffs{1},exp_r2{1}] = find_slope(baseline_sub_Tbinned_F,position,tasks);
    
    %label lap as rewarded(1) or unrewarded (0)
    rewarded(1:size(all_tasks)) = 1;
    rewarded(size(baseline_sub_Tbinned_F{1},1)+1:size(baseline_sub_Tbinned_F{1},1)+size(baseline_sub_Tbinned_F{2},1) + size(baseline_sub_Tbinned_F{3},1))= 0;
    
    
    %divide rerewarded and novel trials into early and late laps
    early_late_Tbin_F{1} = [early_late_Tbin_F{1}; (baseline_sub_Tbinned_F{4}(1:10,:)/max(cell2mat(mean_Tbin_F_sub_baseline)))];
    early_late_Tbin_F{2} = [early_late_Tbin_F{2}; baseline_sub_Tbinned_F{4}(end-9:end,:)/max(cell2mat(mean_Tbin_F_sub_baseline))];
    early_late_Tbin_F{3} = [early_late_Tbin_F{3}; baseline_sub_Tbinned_F{5}(1:10,:)/max(cell2mat(mean_Tbin_F_sub_baseline))];
    early_late_Tbin_F{4} = [early_late_Tbin_F{4}; baseline_sub_Tbinned_F{5}(end-9:end,:)/max(cell2mat(mean_Tbin_F_sub_baseline))];

end
% 
%% 2. 115 analysis
clear bin_mean_F Tbin_mean_F baseline_sub_binned_F baseline_sub_Tbinned_F

load('115/115_splitbeh.mat');
load('115/115_plane2_ROIs.mat');
load('115/115_good_behavior.mat');

plane = 2;
tracklength = 200;

tasks = ["rew","no rew-licks","no rew-no licks","re rew", "nov"];

frames{1} = [69:3998];
frames{2} = [4035:7522];
frames{3} = [7600:11824];
frames{4} = [12371:15979];
frames{5} = [16020:21000];

fields = fieldnames(behplane{1,plane});
combined_max_bin_F_sub_base = [];
combined_max_bin_F_sub_base = [];
combined_max_Tbin_F_sub_base = [];
combined_lap_max_F = [];

for task = 1: length(tasks)

    for field = 1: length(fields)-1
        behavior{task}.(fields{field}) = behplane{1,plane}.(fields{field})(frames{task});
    end

    [bin_mean_F{task},Tbin_mean_F{task}, lap_max_F{task},AUC{task}, time2reward{2,task}, postrewardtime{2,task}, num_licks{2,task}] = binFbyPosition(smooth(data.F(frames{task}),7,'sgolay',5), behavior{task}, tracklength, frames{task});

    [max_bin_F{task},max_bin_F{task}(:,2)] = max(bin_mean_F{task},[],2);
    [max_Tbin_F{task},max_Tbin_F{task}(:,2)] = max(Tbin_mean_F{task},[],2);

    baseline_sub_binned_F{task} = bin_mean_F{task} - min(bin_mean_F{task}(:,:),[],2);
    baseline_sub_Tbinned_F{task} = Tbin_mean_F{task} - min(Tbin_mean_F{task}(:,:),[],2);

    [max_bin_F_sub_base{task},max_bin_F_sub_base{task}(:,2)] = max(baseline_sub_binned_F{task},[],2);
%         
      for lap = 1: size(bin_mean_F{task})
        position = 1:5:tracklength;
        position = position(~isnan(baseline_sub_binned_F{task}(lap,:))); 
        base_sub_bin_F_nanless = baseline_sub_binned_F{task}(lap,~isnan(baseline_sub_binned_F{task}(lap,:)));
        
        position = 1:5:tracklength;
        position = position(~isnan(baseline_sub_Tbinned_F{task}(lap,:))); 
        base_sub_Tbin_F_nanless = baseline_sub_Tbinned_F{task}(lap,~isnan(baseline_sub_Tbinned_F{task}(lap,:)));
      end
% 

    

end


mean_bin_F = plot_pupil_means(bin_mean_F, tasks);
mean_Tbin_F = plot_pupil_means(Tbin_mean_F, tasks);

mean_bin_F_sub_baseline = plot_pupil_means(baseline_sub_binned_F, tasks);
mean_Tbin_F_sub_baseline = plot_pupil_means(baseline_sub_Tbinned_F, tasks);

for task = 1:length(tasks)
    combined_bin_F{task} = [combined_bin_F{task};(bin_mean_F{task}/max(cell2mat(mean_bin_F)))];
    combined_Tbin_F{task} = [combined_Tbin_F{task};(Tbin_mean_F{task}/max(cell2mat(mean_Tbin_F)))];
    combined_bin_F_sub_baseline{task} = [combined_bin_F_sub_baseline{task};(baseline_sub_binned_F{task}/max(cell2mat(mean_bin_F_sub_baseline)))];
    combined_Tbin_F_sub_baseline{task} = [combined_Tbin_F_sub_baseline{task};(baseline_sub_Tbinned_F{task}/max(cell2mat(mean_Tbin_F_sub_baseline)))];

    max_Tbin_F_sub_base_raw{2,task} = max(baseline_sub_Tbinned_F{task},[],2);
    max_Tbin_F_sub_base{2,task}(:,1) = max_Tbin_F_sub_base_raw{2,task}/nanmean(max_Tbin_F_sub_base_raw{2,1}); 
    [placeholder, max_Tbin_F_sub_base{2,task}(:,2)] = max(baseline_sub_Tbinned_F{task},[],2);
    max_Tbin_F_sub_base{2,task}(:,3) = max_Tbin_F_sub_base{2,task}(:,1).*max_Tbin_F_sub_base{2,task}(:,2);
    
    combined_max_Tbin_F_sub_base = [combined_max_Tbin_F_sub_base; max_Tbin_F_sub_base{2,task}];
    combined_lap_max_F = [combined_lap_max_F; lap_max_F{task}'];

    
    
    mouse = 2;
    for section=1: size(sections,1)
        section_means(mouse, task, section) = nanmean(nanmean((baseline_sub_binned_F{task}(:,sections{section})/max(cell2mat(mean_bin_F_sub_baseline)))));
        section_Tmeans(mouse, task, section) = nanmean(nanmean((baseline_sub_Tbinned_F{task}(:,sections{section})/max(cell2mat(mean_Tbin_F_sub_baseline)))));
    end
end
    
[lap_slope{2},line_r2{2},exp_coeffs{2},exp_r2{2}] = find_slope(baseline_sub_Tbinned_F,position,tasks);


 early_late_Tbin_F{1} = [early_late_Tbin_F{1}; (baseline_sub_Tbinned_F{4}(1:10,:)/max(cell2mat(mean_Tbin_F_sub_baseline)))];
early_late_Tbin_F{2} = [early_late_Tbin_F{2}; baseline_sub_Tbinned_F{4}(end-9:end,:)/max(cell2mat(mean_Tbin_F_sub_baseline))];
early_late_Tbin_F{3} = [early_late_Tbin_F{3}; baseline_sub_Tbinned_F{5}(1:10,:)/max(cell2mat(mean_Tbin_F_sub_baseline))];
early_late_Tbin_F{4} = [early_late_Tbin_F{4}; baseline_sub_Tbinned_F{5}(end-9:end,:)/max(cell2mat(mean_Tbin_F_sub_baseline))];

%% 3. 265 analysis
clear bin_mean_F Tbin_mean_F baseline_sub_binned_F baseline_sub_Tbinned_F

load('265/265_splitbeh.mat');
load('265/onlycellsmanualF.mat');
load('265/265_good_behavior.mat');

behplane{1,1}.lick = behplane{1,1}.velocity;

plane = 1;
tracklength = 200;

tasks = ["rew","no rew-licking","no rew-no lick","re rew", "nov"];

frames{1} = [1:3184];
frames{2} = [3190:7191];
frames{3} = [7194:8480];
frames{4} = [8540:12366];
frames{5} = [12527:20000];

fields = fieldnames(behplane{1,plane});
combined_max_bin_F_sub_base = [];
combined_max_bin_F_sub_base = [];
combined_max_Tbin_F_sub_base = [];
combined_lap_max_F = [];
norm_Tbin_F_sub_baseline = {[],[],[],[],[]};


for task = 1: length(tasks)

    for field = 1: length(fields)-1
        behavior{task}.(fields{field}) = behplane{1,plane}.(fields{field})(frames{task});
    end

    [bin_mean_F{task},Tbin_mean_F{task}, lap_max_F{task},AUC{task}, time2reward{3,task}, postrewardtime{3,task}, num_licks{3,task}] = binFbyPosition(smooth(data.F(frames{task}),7,'sgolay',5), behavior{task}, tracklength, frames{task});

    [max_bin_F{task},max_bin_F{task}(:,2)] = max(bin_mean_F{task},[],2);
    [max_Tbin_F{task},max_Tbin_F{task}(:,2)] = max(Tbin_mean_F{task},[],2);

    baseline_sub_binned_F{task} = bin_mean_F{task} - min(bin_mean_F{task}(:,:),[],2);
    baseline_sub_Tbinned_F{task} = Tbin_mean_F{task} - min(Tbin_mean_F{task}(:,:),[],2);


    for lap = 1: size(bin_mean_F{task})
        position = 1:5:tracklength;
        position = position(~isnan(baseline_sub_binned_F{task}(lap,:))); 
        base_sub_bin_F_nanless = baseline_sub_binned_F{task}(lap,~isnan(baseline_sub_binned_F{task}(lap,:)));
        
        position = 1:5:tracklength;
        position = position(~isnan(baseline_sub_Tbinned_F{task}(lap,:))); 
        base_sub_Tbin_F_nanless = baseline_sub_Tbinned_F{task}(lap,~isnan(baseline_sub_Tbinned_F{task}(lap,:)));
    end
 
end


mean_bin_F = plot_pupil_means(bin_mean_F, tasks);
mean_Tbin_F = plot_pupil_means(Tbin_mean_F, tasks);

mean_bin_F_sub_baseline = plot_pupil_means(baseline_sub_binned_F, tasks);
mean_Tbin_F_sub_baseline = plot_pupil_means(baseline_sub_Tbinned_F, tasks);

for task = 1:length(tasks)
    combined_bin_F{task} = [combined_bin_F{task};(bin_mean_F{task}/max(cell2mat(mean_bin_F)))];
    combined_Tbin_F{task} = [combined_Tbin_F{task};(Tbin_mean_F{task}/max(cell2mat(mean_Tbin_F)))];
    combined_bin_F_sub_baseline{task} = [combined_bin_F_sub_baseline{task};(baseline_sub_binned_F{task}/max(cell2mat(mean_bin_F_sub_baseline)))];
    combined_Tbin_F_sub_baseline{task} = [combined_Tbin_F_sub_baseline{task};(baseline_sub_Tbinned_F{task}/max(cell2mat(mean_Tbin_F_sub_baseline)))];
    
    max_Tbin_F_sub_base_raw{3,task} = max(baseline_sub_Tbinned_F{task},[],2);
    max_Tbin_F_sub_base{3,task}(:,1) = max_Tbin_F_sub_base_raw{3,task}/nanmean(max_Tbin_F_sub_base_raw{3,1});     
    [placeholder, max_Tbin_F_sub_base{3,task}(:,2)] = max(baseline_sub_Tbinned_F{task},[],2);
    max_Tbin_F_sub_base{3,task}(:,3) = max_Tbin_F_sub_base{3,task}(:,1).*max_Tbin_F_sub_base{3,task}(:,2);
    
    combined_max_Tbin_F_sub_base = [combined_max_Tbin_F_sub_base; max_Tbin_F_sub_base{3,task}];
    

    
    combined_lap_max_F = [combined_lap_max_F; lap_max_F{task}'];

    
    mouse = 3;
    for section=1: size(sections,1)
        section_means(mouse, task, section) = nanmean(nanmean((baseline_sub_binned_F{task}(:,sections{section})/max(cell2mat(mean_bin_F_sub_baseline)))));
        section_Tmeans(mouse, task, section) = nanmean(nanmean((baseline_sub_Tbinned_F{task}(:,sections{section})/max(cell2mat(mean_Tbin_F_sub_baseline)))));
    end
end

    [lap_slope{3},line_r2{3},exp_coeffs{3},exp_r2{3}] = find_slope(baseline_sub_Tbinned_F,position,tasks);

early_late_Tbin_F{1} = [early_late_Tbin_F{1}; (baseline_sub_Tbinned_F{4}(1:round(size(baseline_sub_Tbinned_F{4},1)/2),:)/max(cell2mat(mean_Tbin_F_sub_baseline)))];
early_late_Tbin_F{2} = [early_late_Tbin_F{2}; baseline_sub_Tbinned_F{4}(round(size(baseline_sub_Tbinned_F{4},1)/2)+1:end,:)/max(cell2mat(mean_Tbin_F_sub_baseline))];
early_late_Tbin_F{3} = [early_late_Tbin_F{3}; baseline_sub_Tbinned_F{5}(1:round(size(baseline_sub_Tbinned_F{4},1)/2),:)/max(cell2mat(mean_Tbin_F_sub_baseline))];
early_late_Tbin_F{4} = [early_late_Tbin_F{4}; baseline_sub_Tbinned_F{5}(round(size(baseline_sub_Tbinned_F{4},1)/2)+1:end,:)/max(cell2mat(mean_Tbin_F_sub_baseline))];y_late_Tbin_F{4} = [early_late_Tbin_F{4}; baseline_sub_Tbinned_F{5}(round(size(baseline_sub_Tbinned_F{4},1)/2)+1:end,:)];

%% 4. 223 analysis

clear bin_mean_F Tbin_mean_F baseline_sub_binned_F baseline_sub_Tbinned_F

load('223/223_splitbeh.mat');
load('223/2ROIs_ROIs.mat');
load('223/223_good_behavior.mat');

behplane{1,1}.lick = behplane{1,1}.velocity;


plane = 1;
tracklength = 200;

tasks = ["rew","no rew-licking","no rew-no lick"];

frames{1} = [1:3683];
frames{2} = [3687:6260]; 
frames{3} = [6265:11360];


fields = fieldnames(behplane{1,plane});

for roi=1: size(data.F,2)
    combined_max_bin_F_sub_base = [];
    combined_max_bin_F_sub_base = [];
    combined_max_Tbin_F_sub_base = [];
    combined_lap_max_F = [];


    for task = 1: length(tasks)

        for field = 1: length(fields)-1
            behavior{task}.(fields{field}) = behplane{1,plane}.(fields{field})(frames{task});
        end

        [bin_mean_F{task},Tbin_mean_F{task}, lap_max_F{task},AUC{task}, time2reward{3+roi,task}, postrewardtime{3+roi,task}, num_licks{3+roi,task}] = binFbyPosition(smooth(data.F(frames{task},roi),7,'sgolay',5), behavior{task}, tracklength, frames{task});

        [max_bin_F{task},max_bin_F{task}(:,2)] = max(bin_mean_F{task},[],2);
        [max_Tbin_F{task},max_Tbin_F{task}(:,2)] = max(Tbin_mean_F{task},[],2);

        baseline_sub_binned_F{task} = bin_mean_F{task} - min(bin_mean_F{task}(:,:),[],2);
        baseline_sub_Tbinned_F{task} = Tbin_mean_F{task} - min(Tbin_mean_F{task}(:,:),[],2);


        for lap = 1: size(bin_mean_F{task})
            position = 1:5:tracklength;
            position = position(~isnan(baseline_sub_binned_F{task}(lap,:))); 
            base_sub_bin_F_nanless = baseline_sub_binned_F{task}(lap,~isnan(baseline_sub_binned_F{task}(lap,:)));

            position = 1:5:tracklength;
            position = position(~isnan(baseline_sub_Tbinned_F{task}(lap,:))); 
            base_sub_Tbin_F_nanless = baseline_sub_Tbinned_F{task}(lap,~isnan(baseline_sub_Tbinned_F{task}(lap,:)));
        end
       
    end


    mean_bin_F = plot_pupil_means(bin_mean_F, tasks);
    mean_Tbin_F = plot_pupil_means(Tbin_mean_F, tasks);

    mean_bin_F_sub_baseline = plot_pupil_means(baseline_sub_binned_F, tasks);
    mean_Tbin_F_sub_baseline = plot_pupil_means(baseline_sub_Tbinned_F, tasks);

    for task = 1:length(tasks)
        combined_bin_F{task} = [combined_bin_F{task};(bin_mean_F{task}/max(cell2mat(mean_bin_F)))];
        combined_Tbin_F{task} = [combined_Tbin_F{task};(Tbin_mean_F{task}/max(cell2mat(mean_Tbin_F)))];
        combined_bin_F_sub_baseline{task} = [combined_bin_F_sub_baseline{task};(baseline_sub_binned_F{task}/max(cell2mat(mean_bin_F_sub_baseline)))];
        combined_Tbin_F_sub_baseline{task} = [combined_Tbin_F_sub_baseline{task};(baseline_sub_Tbinned_F{task}/max(cell2mat(mean_Tbin_F_sub_baseline)))];

        max_Tbin_F_sub_base_raw{3+roi,task} = max(baseline_sub_Tbinned_F{task},[],2);
        max_Tbin_F_sub_base{3+roi,task}(:,1) = max_Tbin_F_sub_base_raw{3+roi,task}/nanmean(max_Tbin_F_sub_base_raw{3+roi,1});         
        [placeholder, max_Tbin_F_sub_base{3+roi,task}(:,2)] = max(baseline_sub_Tbinned_F{task},[],2);
        max_Tbin_F_sub_base{3+roi,task}(:,3) = max_Tbin_F_sub_base{3+roi,task}(:,1).*max_Tbin_F_sub_base{3+roi,task}(:,2);

        combined_max_Tbin_F_sub_base = [combined_max_Tbin_F_sub_base; max_Tbin_F_sub_base{3+roi,task}];


        combined_lap_max_F = [combined_lap_max_F; lap_max_F{task}'];
        
        mouse = 3+roi;
        for section=1: size(sections,1)
            section_means(mouse, task, section) = nanmean(nanmean((baseline_sub_binned_F{task}(:,sections{section})/max(cell2mat(mean_bin_F_sub_baseline)))));
            section_Tmeans(mouse, task, section) = nanmean(nanmean((baseline_sub_Tbinned_F{task}(:,sections{section})/max(cell2mat(mean_Tbin_F_sub_baseline)))));
        end
    end

    [lap_slope{3+roi},line_r2{3+roi},exp_coeffs{3+roi},exp_r2{3+roi}] = find_slope(baseline_sub_Tbinned_F,position,tasks);

end

%% 5. 253 analysis
clear bin_mean_F Tbin_mean_F baseline_sub_binned_F baseline_sub_Tbinned_F

load('253/253_092821_splitbeh.mat');
load('253/253_combined_ROIs.mat');
load('253/253_good_behavior.mat');

plane = 2;
tracklength = 200;

tasks = ["rew","no rew-licking","no rew-no lick", "re-rew", "novel"];

frames{1} = [5:3950];
frames{2} = [4001:4750]; 
frames{3} = [4760:8828];
frames{4} = [9125:12990];
frames{5} = [13040:18000]; 


fields = fieldnames(behplane{1,plane});
combined_max_bin_F_sub_base = [];
combined_max_bin_F_sub_base = [];
combined_max_Tbin_F_sub_base = [];
combined_lap_max_F = [];


for task = 1: length(tasks)

    for field = 1: length(fields)-1
        behavior{task}.(fields{field}) = behplane{1,plane}.(fields{field})(frames{task});
    end

    [bin_mean_F{task},Tbin_mean_F{task}, lap_max_F{task},AUC{task}, time2reward{6,task}, postrewardtime{6,task}, num_licks{6,task}] = binFbyPosition(smooth(data.F(frames{task}),7,'sgolay',5), behavior{task}, tracklength, frames{task});

    [max_bin_F{task},max_bin_F{task}(:,2)] = max(bin_mean_F{task},[],2);
    [max_Tbin_F{task},max_Tbin_F{task}(:,2)] = max(Tbin_mean_F{task},[],2);

    baseline_sub_binned_F{task} = bin_mean_F{task} - min(bin_mean_F{task}(:,:),[],2);
    baseline_sub_Tbinned_F{task} = Tbin_mean_F{task} - min(Tbin_mean_F{task}(:,:),[],2);


    for lap = 1: size(bin_mean_F{task})
        position = 1:5:tracklength;
        position = position(~isnan(baseline_sub_binned_F{task}(lap,:))); 
        base_sub_bin_F_nanless = baseline_sub_binned_F{task}(lap,~isnan(baseline_sub_binned_F{task}(lap,:)));
        
        position = 1:5:tracklength;
        position = position(~isnan(baseline_sub_Tbinned_F{task}(lap,:))); 
        base_sub_Tbin_F_nanless = baseline_sub_Tbinned_F{task}(lap,~isnan(baseline_sub_Tbinned_F{task}(lap,:)));
    end

end


mean_bin_F = plot_pupil_means(bin_mean_F, tasks);
mean_Tbin_F = plot_pupil_means(Tbin_mean_F, tasks);

mean_bin_F_sub_baseline = plot_pupil_means(baseline_sub_binned_F, tasks);
mean_Tbin_F_sub_baseline = plot_pupil_means(baseline_sub_Tbinned_F, tasks);

for task = 1:length(tasks)
    combined_bin_F{task} = [combined_bin_F{task};(bin_mean_F{task}/max(cell2mat(mean_bin_F)))];
    combined_Tbin_F{task} = [combined_Tbin_F{task};(Tbin_mean_F{task}/max(cell2mat(mean_Tbin_F)))];
    combined_bin_F_sub_baseline{task} = [combined_bin_F_sub_baseline{task};(baseline_sub_binned_F{task}/max(cell2mat(mean_bin_F_sub_baseline)))];
    combined_Tbin_F_sub_baseline{task} = [combined_Tbin_F_sub_baseline{task};(baseline_sub_Tbinned_F{task}/max(cell2mat(mean_Tbin_F_sub_baseline)))];

    max_Tbin_F_sub_base_raw{6,task} = max(baseline_sub_Tbinned_F{task},[],2);
    max_Tbin_F_sub_base{6,task}(:,1) = max_Tbin_F_sub_base_raw{6,task}/nanmean(max_Tbin_F_sub_base_raw{6,1}); 
    [placeholder, max_Tbin_F_sub_base{6,task}(:,2)] = max(baseline_sub_Tbinned_F{task},[],2);
    max_Tbin_F_sub_base{6,task}(:,3) = max_Tbin_F_sub_base{6,task}(:,1).*max_Tbin_F_sub_base{6,task}(:,2);

    combined_max_Tbin_F_sub_base = [combined_max_Tbin_F_sub_base; max_Tbin_F_sub_base{6,task}];
    combined_lap_max_F = [combined_lap_max_F; lap_max_F{task}'];

    mouse = 6;
    for section=1: size(sections,1)
        section_means(mouse, task, section) = nanmean(nanmean((baseline_sub_binned_F{task}(:,sections{section})/max(cell2mat(mean_bin_F_sub_baseline)))));
        section_Tmeans(mouse, task, section) = nanmean(nanmean((baseline_sub_Tbinned_F{task}(:,sections{section})/max(cell2mat(mean_Tbin_F_sub_baseline)))));
    end
end

[lap_slope{6},line_r2{6},exp_coeffs{6},exp_r2{6}] = find_slope(baseline_sub_Tbinned_F,position,tasks);

early_late_Tbin_F{1} = [early_late_Tbin_F{1}; (baseline_sub_Tbinned_F{4}(1:10,:)/max(cell2mat(mean_Tbin_F_sub_baseline)))];
early_late_Tbin_F{2} = [early_late_Tbin_F{2}; baseline_sub_Tbinned_F{4}(end-9:end,:)/max(cell2mat(mean_Tbin_F_sub_baseline))];
early_late_Tbin_F{3} = [early_late_Tbin_F{3}; baseline_sub_Tbinned_F{5}(1:10,:)/max(cell2mat(mean_Tbin_F_sub_baseline))];
early_late_Tbin_F{4} = [early_late_Tbin_F{4}; baseline_sub_Tbinned_F{5}(end-9:end,:)/max(cell2mat(mean_Tbin_F_sub_baseline))];


%% 6. 255 analysis
clear bin_mean_F Tbin_mean_F baseline_sub_binned_F baseline_sub_Tbinned_F

load('255/093021/255_093021_splitbeh.mat');
load('255/093021/255_combinedROI_ROIs.mat');
load('255/093021/255_good_behavior.mat');

plane = 2;
tracklength = 200;

tasks = ["rew","no rew-licking","no rew-no lick", "re-rew", "novel"];

frames{1} = [125:3936];
frames{2} = [4093:5015]; 
frames{3} = [5020:8076];
frames{4} = [8330:11860];
frames{5} = [12060:16000]; 


fields = fieldnames(behplane{1,plane});
combined_max_bin_F_sub_base = [];
combined_max_bin_F_sub_base = [];
combined_max_Tbin_F_sub_base = [];
combined_lap_max_F = [];

for task = 1: length(tasks)

    for field = 1: length(fields)-1
        behavior{task}.(fields{field}) = behplane{1,plane}.(fields{field})(frames{task});
    end

    [bin_mean_F{task},Tbin_mean_F{task}, lap_max_F{task},AUC{task}, time2reward{7,task}, postrewardtime{7,task}, num_licks{7,task}] = binFbyPosition(smooth(data.F(frames{task}),7,'sgolay',5), behavior{task}, tracklength, frames{task});

    [max_bin_F{task},max_bin_F{task}(:,2)] = max(bin_mean_F{task},[],2);
    [max_Tbin_F{task},max_Tbin_F{task}(:,2)] = max(Tbin_mean_F{task},[],2);

    baseline_sub_binned_F{task} = bin_mean_F{task} - min(bin_mean_F{task}(:,:),[],2);
    baseline_sub_Tbinned_F{task} = Tbin_mean_F{task} - min(Tbin_mean_F{task}(:,:),[],2);


      for lap = 1: size(bin_mean_F{task})
        position = 1:5:tracklength;
        position = position(~isnan(baseline_sub_binned_F{task}(lap,:))); 
        base_sub_bin_F_nanless = baseline_sub_binned_F{task}(lap,~isnan(baseline_sub_binned_F{task}(lap,:)));
        
        position = 1:5:tracklength;
        position = position(~isnan(baseline_sub_Tbinned_F{task}(lap,:))); 
        base_sub_Tbin_F_nanless = baseline_sub_Tbinned_F{task}(lap,~isnan(baseline_sub_Tbinned_F{task}(lap,:)));
      end
    
end


mean_bin_F = plot_pupil_means(bin_mean_F, tasks);
mean_Tbin_F = plot_pupil_means(Tbin_mean_F, tasks);

mean_bin_F_sub_baseline = plot_pupil_means(baseline_sub_binned_F, tasks);
mean_Tbin_F_sub_baseline = plot_pupil_means(baseline_sub_Tbinned_F, tasks);

for task = 1:length(tasks)
    combined_bin_F{task} = [combined_bin_F{task};(bin_mean_F{task}/max(cell2mat(mean_bin_F)))];
    combined_Tbin_F{task} = [combined_Tbin_F{task};(Tbin_mean_F{task}/max(cell2mat(mean_Tbin_F)))];
    combined_bin_F_sub_baseline{task} = [combined_bin_F_sub_baseline{task};(baseline_sub_binned_F{task}/max(cell2mat(mean_bin_F_sub_baseline)))];
    combined_Tbin_F_sub_baseline{task} = [combined_Tbin_F_sub_baseline{task};(baseline_sub_Tbinned_F{task}/max(cell2mat(mean_Tbin_F_sub_baseline)))];

    max_Tbin_F_sub_base_raw{7,task} = max(baseline_sub_Tbinned_F{task},[],2);
    max_Tbin_F_sub_base{7,task}(:,1) = max_Tbin_F_sub_base_raw{7,task}/nanmean(max_Tbin_F_sub_base_raw{7,1}); 
    [placeholder, max_Tbin_F_sub_base{7,task}(:,2)] = max(baseline_sub_Tbinned_F{task},[],2);
    max_Tbin_F_sub_base{7,task}(:,3) = max_Tbin_F_sub_base{7,task}(:,1).*max_Tbin_F_sub_base{7,task}(:,2);

    combined_max_Tbin_F_sub_base = [combined_max_Tbin_F_sub_base; max_Tbin_F_sub_base{7,task}];
    combined_lap_max_F = [combined_lap_max_F; lap_max_F{task}'];

    
    mouse = 7;
    for section=1: size(sections,1)
        section_means(mouse, task, section) = nanmean(nanmean((baseline_sub_binned_F{task}(:,sections{section})/max(cell2mat(mean_bin_F_sub_baseline)))));
        section_Tmeans(mouse, task, section) = nanmean(nanmean((baseline_sub_Tbinned_F{task}(:,sections{section})/max(cell2mat(mean_Tbin_F_sub_baseline)))));
    end
end

[lap_slope{7},line_r2{7},exp_coeffs{7},exp_r2{7}] = find_slope(baseline_sub_Tbinned_F,position,tasks);
    

 early_late_Tbin_F{1} = [early_late_Tbin_F{1}; (baseline_sub_Tbinned_F{4}(1:10,:)/max(cell2mat(mean_Tbin_F_sub_baseline)))];
early_late_Tbin_F{2} = [early_late_Tbin_F{2}; baseline_sub_Tbinned_F{4}(end-9:end,:)/max(cell2mat(mean_Tbin_F_sub_baseline))];
early_late_Tbin_F{3} = [early_late_Tbin_F{3}; baseline_sub_Tbinned_F{5}(1:10,:)/max(cell2mat(mean_Tbin_F_sub_baseline))];
early_late_Tbin_F{4} = [early_late_Tbin_F{4}; baseline_sub_Tbinned_F{5}(end-9:end,:)/max(cell2mat(mean_Tbin_F_sub_baseline))];


%% 7. plot means and 95% CI for sectioned data

for section = 1: size(sections,1)
    %calculate SEM for sectioned data
    sectioned_SEM(:,section) = nanstd(section_means(:,:,section))./sqrt([7,7,7,5,5]);

    %calculate CI95 and run statistics on sectioned data
    for task = 1: size(combined_Tbin_F,2)
        CI95(task,:) = tinv([0.025 0.975], length(section_means(:,task,section))-1);
        sectioned_CI95(task,section,:) = bsxfun(@times,sectioned_SEM(task,section), CI95(task,:));
        sectioned_mean_and_CI95(task,section,:) = nanmean(section_means(:,task,section)) + sectioned_CI95(task,section,:);
        
        [significance(task,section),section_p(task,section)] = ttest(section_means(:,1,section),section_means(:,task,section))
    end
    make_boxplot(squeeze(section_means(:,1:3,section)))
end

figure;
errorbar(squeeze(nanmean(section_means(:,1:3,:)))',sectioned_CI95(1:3,:,1)', '-o')
xlim([0.5,3.5])

%% 8. Plot mean time and position binned activity of all mice 

task_time = {[],[],[],[],[],[],[]};
post_reward_t = task_time;

%subytsvy baseline and calculate the mean time to reward and the mean time
%after reward
for task = 1: size(combined_Tbin_F,2)
    for mouse = 1:(size(time2reward,1))
        task_time{task} = [task_time{task}, time2reward{mouse,task}];
        post_reward_t{task} = [post_reward_t{task}, postrewardtime{mouse,task}];
    end
    mean_t(task) = nanmean(task_time{task});
    mean_post_r_t(task) = nanmean(post_reward_t{task});
    
    min_sub_bin_F{task} = combined_bin_F{task} - min(nanmean(combined_bin_F{task}));
    min_sub_Tbin_F{task} = combined_Tbin_F{task} - min(nanmean(combined_Tbin_F{task}));
    min_sub_bin_F_sub_b{task} = combined_bin_F_sub_baseline{task} - min(nanmean(combined_bin_F_sub_baseline{task}));
    min_sub_Tbin_F_sub_b{task} = combined_Tbin_F_sub_baseline{task} - min(nanmean(combined_Tbin_F_sub_baseline{task}));
    
    early_late_Tbin_F_min_sub{task} = early_late_Tbin_F{task} - min(nanmean(early_late_Tbin_F{task}));
end


%plot time binned and position binned data of all mice for all tasks
position = 1:5:200;
tasks = ["rew","no rew-licks","no rew- no licks",'rereward','novel'];
plot_pupil_means(min_sub_bin_F, tasks)

plot_pupil_means(min_sub_Tbin_F, tasks)

% plot_laps_with_mean(position, combined_Tbin_F_sub_baseline, tasks, [0, 1.5])

plot_pupil_means(min_sub_bin_F_sub_b,tasks)

y_curve = plot_pupil_means(min_sub_Tbin_F_sub_b, tasks)

%plot time binned data with line for reward delivery
position = 1:5:200;
tasks = ["rerew early","rerew late","novel early",'novel late'];
plot_pupil_means(early_late_Tbin_F_min_sub, tasks)
hold on
line([-6 -6],[0.8,-.2])

%% 9. Align binned data with removal of reward

%change 0 values to NaN
% AUC_mean(AUC_mean== 0) = NaN;
% AUC_mean_bin_F(AUC_mean_bin_F == 0) = NaN;
% AUC_mean_Tbin_F(AUC_mean_Tbin_F == 0) = NaN;

%initialize variables 
% aligned_AUC_Tbin = NaN(size(combined_AUC_Tbin,1), 50);
aligned_Max_Tbin = NaN(size(section_means,1), 50);
aligned_slope = NaN(size(section_means,1), 50);
aligned_coeffs = NaN(size(section_means,1), 50);
switch_aligned_Max_Tbin = NaN(size(section_means,1), 50);;
switch_aligned_slope = switch_aligned_Max_Tbin;
switch_aligned_coeffs = switch_aligned_Max_Tbin;
rereward_slope = switch_aligned_Max_Tbin;


for axon = 1: size(section_means,1)
    % calculate the slope x max 
    slope_x_max(axon,:) = nanmean(lap_slope{axon}(1,~isnan(lap_slope{axon}(1,:))).*max_Tbin_F_sub_base{axon,1}(:,1)');
    for task = 1: size(lap_slope{axon},1)
        
        exp_coeffs{axon}(task,1:sum(~isnan(lap_slope{axon}(task,:)))) = (lap_slope{axon}(task,~isnan(lap_slope{axon}(task,:))).*max_Tbin_F_sub_base{axon,task}(:,1)')/slope_x_max(axon);
        
        if ~isempty(max_Tbin_F_sub_base{axon, task})
            max_mean_Tbin_F(axon,task) = nanmean(max_Tbin_F_sub_base{axon, task}(:,1));
            mean_lap_slope(axon,task) = nanmean(lap_slope{axon}(task,:));
            mean_lap_coeff(axon,task) = nanmean(exp_coeffs{axon}(task,:));
        end
    
        max_num_licks(axon,task) = max(num_licks{axon,task});
    end
    
    %normalize licking frequency
    for task =1: size(lap_slope{axon},1)
        norm_num_licks{axon,task} = num_licks{axon,task}./max(max_num_licks(axon,1:3));
        norm_num_licks{axon,task}(norm_num_licks{axon,task} == NaN) = 0;
    end
    
    %align max, slope and slope x max to reward removal
%     aligned_AUC_Tbin(axon,21:20+length(combined_AUC_Tbin{axon,3})) = combined_AUC_Tbin{axon,3}(:);
    aligned_Max_Tbin(axon,21:20+length(max_Tbin_F_sub_base{axon,3}(:,1))) = max_Tbin_F_sub_base{axon,3}(:,1);
    aligned_slope(axon,21:20+sum(~isnan(lap_slope{axon}(3,:)))) = lap_slope{axon}(3,~isnan(lap_slope{axon}(3,:)));
    aligned_coeffs(axon,21:20+sum(~isnan(exp_coeffs{axon}(3,:)))) = exp_coeffs{axon}(3,~isnan(exp_coeffs{axon}(3,:)));
    
%     aligned_AUC_Tbin(axon,21-length(combined_AUC_Tbin{axon,2}):20) = combined_AUC_Tbin{axon,2}(:);
    aligned_Max_Tbin(axon,21-length(max_Tbin_F_sub_base{axon,2}(:,1)):20) = max_Tbin_F_sub_base{axon,2}(:,1);
    aligned_slope(axon,21-sum(~isnan(lap_slope{axon}(2,:))):20) = lap_slope{axon}(2,~isnan(lap_slope{axon}(2,:)));
    aligned_coeffs(axon,21-sum(~isnan(exp_coeffs{axon}(2,:))):20) = exp_coeffs{axon}(2,~isnan(exp_coeffs{axon}(2,:)));

%     aligned_AUC_Tbin(axon,1) =  combined_AUC_Tbin{axon,1}(end);
    aligned_Max_Tbin(axon,1) =  max_Tbin_F_sub_base{axon,1}(end,1);
    aligned_slope(axon,1) = lap_slope{axon}(1, sum(~isnan(lap_slope{axon}(1,:))));
    aligned_coeffs(axon,1) = exp_coeffs{axon}(1, sum(~isnan(exp_coeffs{axon}(1,:))));
 
    switch_aligned_Max_Tbin(axon,1:8) = max_Tbin_F_sub_base{axon,1}(end-7:end,1);
    noreward_max_Tbin = [max_Tbin_F_sub_base{axon,2}(:,1); max_Tbin_F_sub_base{axon,3}(:,1)];
    switch_aligned_Max_Tbin(axon,9:20) = noreward_max_Tbin(1:12);
    switch_aligned_Max_Tbin(axon,1) =  max_Tbin_F_sub_base{axon,1}(end,1);
    
    switch_aligned_slope(axon,1:8) = lap_slope{axon}(1,find(~isnan(lap_slope{axon}(1,:)),1,'last')-7:find(~isnan(lap_slope{axon}(1,:)),1,'last'));
    noreward_slope = [lap_slope{axon}(2,~isnan(lap_slope{axon}(2,:))), lap_slope{axon}(3,:)];
    switch_aligned_slope(axon,9:20) = noreward_slope(1:12);
    
    switch_aligned_coeffs(axon,1:8) = exp_coeffs{axon}(1,find(~isnan(exp_coeffs{axon}(1,:)),1,'last')-7:find(~isnan(exp_coeffs{axon}(1,:)),1,'last'));
    noreward_coeffs = [exp_coeffs{axon}(2,~isnan(exp_coeffs{axon}(2,:))), exp_coeffs{axon}(3,:)];
    switch_aligned_coeffs(axon,9:20) = noreward_coeffs(1:12);
    
    
    switch_aligned_num_licks(axon,1:8) = norm_num_licks{axon,1}(find(~isnan(norm_num_licks{axon,1}(:)),1,'last')-7:find(~isnan(norm_num_licks{axon,1}(:)),1,'last'));
    noreward_num_licks = [norm_num_licks{axon,2}(~isnan(norm_num_licks{axon,2}(:))), norm_num_licks{axon,3}(:)'];
    switch_aligned_num_licks(axon,9:20) = noreward_num_licks(1:12);
    
   rereward_slope(axon,1:12) = noreward_coeffs(1,find(isnan(noreward_coeffs),1)-12:find(isnan(noreward_coeffs),1)-1);
   if size(lap_slope{axon},1) == 5
       rereward_slope(axon,13:22) = exp_coeffs{axon}(4,1:10);
   end
   
   
end



% make boxplots of max, slope and slope x max values for each task
% max_mean_Tbin_F(AUC_mean_Tbin_F == 0) = NaN;
max_mean_Tbin_F(max_mean_Tbin_F == 0) = NaN;
make_boxplot(max_mean_Tbin_F(:,1:3))
make_boxplot(max_mean_Tbin_F([1:3,6,7],[1,4]))
max_mean_Tbin_F(max_mean_Tbin_F == 0) = NaN;
mean_max_tasks = nanmean(max_mean_Tbin_F);
SEM_max_tasks = nanstd(max_mean_Tbin_F)./sqrt([7,7,7,5,5]);

make_boxplot(mean_lap_slope(:,1:3))
make_boxplot(mean_lap_slope([1:3,6,7],[1,4]))
mean_lap_slope(mean_lap_slope == 0) = NaN;
mean_slope_tasks = nanmean(mean_lap_slope);
SEM_slope_tasks = nanstd(mean_lap_slope)./sqrt([7,7,7,5,5]);

make_boxplot(mean_lap_coeff(:,1:3))
make_boxplot(mean_lap_coeff([1:3,6,7],[1,4]))
mean_lap_coeff(mean_lap_coeff == 0) = NaN;
mean_coeff_tasks = nanmean(mean_lap_coeff);
SEM_coeff_tasks = nanstd(mean_lap_coeff)./sqrt([7,7,7,5,5]);

%ginf CI95 for aligned slope, max and slope x max
n = sum(~isnan(max_mean_Tbin_F));
for task = 1: length(n)
    CI95(task,:) = tinv([0.025 0.975], n(task)-1);
    max_CI95(task,:) = bsxfun(@times,SEM_max_tasks(task), CI95(task,:));
    max_mean_and_CI95(task,:) = mean_max_tasks(task) + max_CI95(task,:);
    
    slope_CI95(task,:) = bsxfun(@times,SEM_slope_tasks(task), CI95(task,:));
    slope_mean_and_CI95(task,:) = mean_slope_tasks(task) + slope_CI95(task,:);
    
    coeff_CI95(task,:) = bsxfun(@times,SEM_coeff_tasks(task), CI95(task,:));
    coeff_mean_and_CI95(task,:) = mean_coeff_tasks(task) + coeff_CI95(task,:);
end

% find SEM of slope, max and slope x max
for lap = 1:size(aligned_Max_Tbin,2)
%     SEM_AUC_Tbin(lap) = nanstd(aligned_AUC_Tbin(:,lap))/sqrt(sum(~isnan(aligned_AUC_Tbin(:,lap))));
    SEM_Max_Tbin(lap) = nanstd(aligned_Max_Tbin(:,lap))/sqrt(sum(~isnan(aligned_Max_Tbin(:,lap))));
    switch_SEM_max_Tbin(lap) = nanstd(switch_aligned_Max_Tbin(:,lap))/sqrt(sum(~isnan(switch_aligned_Max_Tbin(:,lap))));
    switch_SEM_slope(lap) = nanstd(switch_aligned_slope(:,lap))/sqrt(sum(~isnan(switch_aligned_slope(:,lap))));
    SEM_slope(lap) = nanstd(aligned_slope(:,lap))/sqrt(sum(~isnan(aligned_slope(:,lap))));
    switch_SEM_coeff(lap) = nanstd(switch_aligned_coeffs(:,lap))/sqrt(sum(~isnan(switch_aligned_coeffs(:,lap))));
    SEM_coeff(lap) = nanstd(aligned_coeffs(:,lap))/sqrt(sum(~isnan(aligned_coeffs(:,lap))));
    SEM_rereward_slope(lap) = nanstd(rereward_slope([1:3,6,7],lap))/sqrt(sum(~isnan(rereward_slope([1:3,6,7],lap))));
    

end
% 
% figure;
% errorbar(nanmean(aligned_AUC_Tbin(:,[1:2,11:30])),SEM_AUC_Tbin([1:2,11:30]), '-o')

%plot aligned slope, max and slope x max 
figure;
errorbar(nanmean(aligned_Max_Tbin(:,[1:2,11:30])),SEM_Max_Tbin([1:2,11:30]), '-o')

figure;
errorbar(nanmean(switch_aligned_Max_Tbin),switch_SEM_max_Tbin, '-o')

figure;
errorbar(nanmean(switch_aligned_slope),switch_SEM_slope, '-o')

figure;
errorbar(nanmean(aligned_slope(:,[1:2,11:30])),SEM_slope([1:2,11:30]), '-o')

figure;
errorbar(nanmean(switch_aligned_coeffs),switch_SEM_coeff, '-o')

figure;
errorbar(nanmean(aligned_coeffs(:,[1:2,11:30])),SEM_coeff([1:2,11:30]), '-o')

figure;
errorbar(nanmean(rereward_slope([1:3,6,7],1:22)),SEM_rereward_slope(1:22), '-o')

figure;
hold on
for axon = 1: size(section_means,1)
    figure;
    hold on
    plot([1:50],switch_aligned_coeffs(axon,:))
end
 
for lap = 1: length(switch_aligned_num_licks)
    SEM_num_licks(lap) = nanstd(switch_aligned_num_licks(:,lap))/sqrt(length(switch_aligned_num_licks(:,lap)));
end

figure;
errorbar(nanmean(switch_aligned_num_licks), SEM_num_licks, '-o')
