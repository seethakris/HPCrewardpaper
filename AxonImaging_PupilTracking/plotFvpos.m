% plot fluorescent data with the animals position for the roi's given
% Chad Heer; Sheffield Lab

function [] = plotFvpos(Fdata, Yposdata, reward, lick, time, roi, smoothby)

% Fdata = fluorescent data loaded in, should be just F, Fc or Fc3
% yposdata = Y_position data 
% reward = reward delivery data
% lick = mouse licking data
% time = time of each data point
% roi = specify which rois to plot F 
% smoothby = specify the span to smooth by

if ~exist('roi', 'var');
    roi = 1;
end

if ~exist('smoothby', 'var');
    smoothby = 0
end

if smoothby == 0;
    for i=1:length(roi)
        figure;
        hold on
        plot(Fdata(:,roi(i))/max(Fdata(:,roi(i))))
        plot(Yposdata/max(Yposdata))
        title(num2str(roi(i)))
    end
    
else
    for i=1:length(roi)
        figure;
        hold on
        plot(time,smooth(Fdata(:,roi(i)), smoothby, 'sgolay',5))
        plot(time,Yposdata/max(Yposdata)-1)
        plot(time, reward)
        plot(time,lick/max(lick)-3)
        
    end
end


        