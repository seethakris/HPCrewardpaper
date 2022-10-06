%%Processes and Bins pupil area data by position, calculates mean velocity, pupil COM distance from mean, blinking ratio, and freezing ratio
%Chad Heer; Sheffield Lab

%INPUTS
%behavior = single plan behavior 
%bins = number of bins 
%idx = frames to bin pupil data over
%good_behavior = single plane output of remove bad behavior scripts
%avg_rew_location = if the task is unrewarded, include the average rew
%location from rewarded laps

%OUTPUTS
%lap_pupil = binned pupil area for each lap (lap x bin)
%beh = contains mean velocity, pupil COM distance from mean, blinking
%ratio, and freezing ratio for each lap
%ypos = contains info used to calculate beh data


function [lap_pupil, beh, ypos] = bin_pupil(behavior, bins, idx, good_behavior, avg_rew_location)

bins = 40;
%remove blinking periods pupil area data
blink_thresh = mean(behavior.blink_0) - 2 * std(behavior.blink_0);
behavior.pupil{1,1}.area(behavior.blink_0 < blink_thresh) = NaN; 
behavior.pupil{1,1}.com(behavior.blink_0 < blink_thresh,:) = NaN; 

%use mouse land script to smooth pupil area and remove blinks
smooth_pupil = smoothPupil(behavior.pupil{1,1}.area);

%interpolate pupil data over entire video
behavior.int_pupil = interp1([1:2:length(smooth_pupil)*2],smooth_pupil, [1:length(smooth_pupil)*2]); %interpolate pupil data for missing frames
behavior.int_blink = interp1([1:2:length(behavior.blink_0)*2],behavior.blink_0, [1:length(behavior.blink_0)*2]); %interpolate pupil data for missing frames
behavior.in_com(:,1) = interp1([1:2:length(behavior.pupil{1,1}.com(:,1))*2],behavior.pupil{1,1}.com(:,1), [1:length(behavior.pupil{1,1}.com(:,1))*2]);
behavior.in_com(:,2) = interp1([1:2:length(behavior.pupil{1,1}.com(:,2))*2],behavior.pupil{1,1}.com(:,2), [1:length(behavior.pupil{1,1}.com(:,2))*2]);

ypos.pupil_com = (behavior.in_com);


%bin position and obtain bin indexes
bin_edges = [0:max(behavior.beh.behplane{1,1}.ybinned)/bins:max(behavior.beh.behplane{1,1}.ybinned)]; 
bin_indexes = discretize(behavior.beh.behplane{1,1}.ybinned, bin_edges);


bin_indexes = bin_indexes(idx);

lap_number = zeros(1,length(behavior.beh.behplane{1,1}.ybinned));
%find lap number 
BW = ones(size(bin_indexes));
BW(find(diff(bin_indexes)<-1)) = 0;
ypos.lap_number(idx) = bwlabel(BW);
lap_number = ypos.lap_number(idx);


behavior.beh.behplane{1,1}.velocity = [diff(behavior.beh.behplane{1,1}.ybinned),0];
behavior.beh.behplane{1,1}.ybinned(behavior.beh.behplane{1,1}.ybinned <= 0.01) = NaN; 
behavior.beh.behplane{1,1}.velocity(behavior.beh.behplane{1,1}.ybinned <= 0.01) = NaN; 
behavior.beh.behplane{1,1}.velocity(behavior.beh.behplane{1,1}.velocity <= -0.005) = NaN; 

lap_number(isnan(behavior.beh.behplane{1,1}.ybinned(idx))) = 0;
   
% 
% use the freezing indexes to remove pupil and position data where the
% animal is not moving
if exist('good_behavior', 'var')
    behavior.int_pupil(good_behavior.good_behavior.freezing_index) = NaN;
    behavior.beh.behplane{1,1}.ybinned(good_behavior.good_behavior.freezing_index) = NaN;
    behavior.beh.behplane{1,1}.lick(good_behavior.good_behavior.freezing_index) = NaN;
    ypos.pupil_com(good_behavior.good_behavior.freezing_index) = NaN;
    behavior.beh.behplane{1,1}.velocity(good_behavior.good_behavior.freezing_index) = NaN;
    ypos.ybinned = behavior.beh.behplane{1,1}.ybinned(good_behavior.good_behavior.good_runs_index(good_behavior.good_behavior.good_runs_index <= max(idx) &  good_behavior.good_behavior.good_runs_index >= min(idx)));
    ypos.lick = behavior.beh.behplane{1,1}.lick(good_behavior.good_behavior.good_runs_index(good_behavior.good_behavior.good_runs_index <= max(idx) &  good_behavior.good_behavior.good_runs_index >= min(idx)));
    ypos.pupil = behavior.int_pupil(good_behavior.good_behavior.good_runs_index(good_behavior.good_behavior.good_runs_index <= max(idx) &  good_behavior.good_behavior.good_runs_index >= min(idx)));
    ypos.velocity = behavior.beh.behplane{1,1}.velocity(good_behavior.good_behavior.good_runs_index(good_behavior.good_behavior.good_runs_index <= max(idx) &  good_behavior.good_behavior.good_runs_index >= min(idx)));
    ypos.lap_number = ypos.lap_number(good_behavior.good_behavior.good_runs_index(good_behavior.good_behavior.good_runs_index <= max(idx) &  good_behavior.good_behavior.good_runs_index >= min(idx)));
else
end

%grab the behavior data for requested frames
behavior.int_pupil = behavior.int_pupil(idx);
behavior.beh.behplane{1,1}.velocity = behavior.beh.behplane{1,1}.velocity(idx);
behavior.beh.behplane{1,1}.ybinned = behavior.beh.behplane{1,1}.ybinned(idx);
behavior.beh.behplane{1,1}.lick = behavior.beh.behplane{1,1}.lick(idx);
behavior.beh.behplane{1,1}.reward = behavior.beh.behplane{1,1}.reward(idx);
ypos.pupil_com = ypos.pupil_com(idx,:);

%calculate the distance from the mean pupil center of mass for each frame
mean_com = nanmean(ypos.pupil_com);
for frame = 1:length(ypos.pupil_com);
    ypos.com_dist_from_mean(frame) = pdist([[mean_com(1) mean_com(2)];[ypos.pupil_com(frame,1) ypos.pupil_com(frame,2)]]);
end

%find the mean velocity of each lap, the end of the track, and the rew
%location
for lap = 1:max(lap_number)
    lap_mean_velocity(lap) = nanmean(behavior.beh.behplane{1,1}.velocity(find(lap_number == lap)));
    
    if exist('avg_rew_location', 'var')
        end_track(lap) = find(behavior.beh.behplane{1,1}.ybinned >= avg_rew_location,1)
        beh.num_licks(lap) = sum(behavior.beh.behplane{1,1}.lick(find(lap_number == lap,1):end_track(lap)) > 1); 
    elseif sum(behavior.beh.behplane{1,1}.reward(lap_number == lap) == 1) >= 1
        rew_frame(lap) = find(behavior.beh.behplane{1,1}.reward == 1 & lap_number == lap, 1);
        beh.rew_location(lap) = behavior.beh.behplane{1,1}.ybinned(rew_frame(lap));
        beh.num_licks(lap) = sum(behavior.beh.behplane{1,1}.lick(find(lap_number == lap,1):rew_frame(lap)) > 1);
    end
end

%bin the pupil area, com distance from mean, and lap_velocity by position
for lap = 1:max(lap_number)
    for bin = 1:bins
        lap_pupil(lap,bin) = nanmean(behavior.int_pupil(lap_number == lap & bin_indexes == bin)); 
        com_mean_dist(lap,bin) = nanmean(ypos.com_dist_from_mean(lap_number == lap & bin_indexes == bin)); 
        ypos.lap_velocity(lap,bin) = nanmean(behavior.beh.behplane{1,1}.velocity(lap_number == lap & bin_indexes == bin));
    end
    if any(behavior.beh.behplane{1,1}.lick(lap_number == lap) >= 0.5)
        lap_licking(lap) = 1;
    else
        lap_licking(lap) = 0;
    end
end



for lap = 1:max(lap_number)
    freezing_ratio(lap) = sum(isnan(behavior.int_pupil(lap_number == lap)))/sum(lap_number == lap);
    beh.blinking_ratio(lap) = sum(behavior.int_blink(lap_number == lap) <= blink_thresh)/sum(lap_number == lap);

    if sum(isnan(lap_pupil(lap,:))) > 1
        lap_pupil(lap,:) = NaN;
        com_mean_dist(lap,:) = NaN;
        lap_licking(lap) = NaN;
        freezing_ratio(lap) = NaN;
        beh.blinking_ratio(lap) = NaN;
        lap_mean_velocity(lap) = NaN; 
    end
end



beh.licking = lap_licking;
beh.mean_velocity = lap_mean_velocity;
beh.freezing_ratio = freezing_ratio; 
beh.lap_com_dist = com_mean_dist;
beh.com_mean_dist = nanmean(com_mean_dist,2); 

mean_rew_lap_pupil = mean(lap_pupil);
SEM_rew_lap_pupil = std(lap_pupil)./sqrt(length(lap_pupil));


