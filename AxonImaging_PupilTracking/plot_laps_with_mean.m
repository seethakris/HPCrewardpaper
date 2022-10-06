%%Plots binned data for each lap in grey and the mean data in black
%Chad Heer; Sheffiel Lab

function [] = plot_laps_with_mean(position, bin_mean_activity,tasks,ylims)
%position = position for each bin
%bin_mean_activity = {task} (lap x binned mean activity)
%tasks = string array listing the order of the tasks
%ylims = y limits 

for task = 1: length(tasks)
    
    %set figure parameters
    figure;
    hold on
    box off
    set(gca,'TickLength',[0 0])
    set(gca,'TickLength',[0 0])
    yticks([min(ylims),mean(ylims),max(ylims)])
    xticks([0,100,200])
    xlabel('Track position(cm)')
    ylabel('Normalized Velocity')
    
    %plot binned data for each lap in grey
    for lap =1:size(bin_mean_activity{task},1)
        plot1 = plot(position, bin_mean_activity{task}(lap,:),'Color',[0.8 0.8 0.8], 'LineWidth', 3);
        plot1.Color(4) = 0.5;
        
    end
    
    %plot mean of binned datat in black
    plot(position,nanmean(bin_mean_activity{task}(:,:)), 'k', 'LineWidth', 3)
    title(tasks(task))
    if ~isempty(ylims)
        ylim(ylims)
    end
end
