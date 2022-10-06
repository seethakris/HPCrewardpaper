%%Plots and calculates the mean bin value of all laps for data in each given task
%Chad Heer; Sheffield Lab

function [mean_lap_pupil] = plot_pupil_means(lap_pupil, tasks)

%lap_pupil = pupil or fluorescence data organized in this format {task}(laps, mean pupil area in each bin)
%tasks = string array listing the order of the tasks

%mean_lap_pupil = the mean of each lap_pupil across the lap dimension 
bins = size(lap_pupil{1},2);
color_seq = ["b" "r" "g" "c" "m" "y" "k"];
figure;
hold on
legend('Location','northwest')
for task = 1: size(tasks,2)
    mean_lap_pupil{task}(:) = nanmean(lap_pupil{task}(:,:));
    SEM_lap_pupil{task}(:) = nanstd(lap_pupil{task}(:,:))./(sqrt(size(lap_pupil{task}(:,:),1)));
    
    
    
    plot([-size(lap_pupil{task},2):-1],mean_lap_pupil{task}(:),color_seq(task), 'LineWidth',2, 'DisplayName',tasks(task))
    patch([[-size(lap_pupil{task},2):-1] fliplr([-size(lap_pupil{task},2):-1])], [(mean_lap_pupil{task}(:)'+SEM_lap_pupil{task}(:)') fliplr(mean_lap_pupil{task}(:)'-SEM_lap_pupil{task}(:)')],color_seq(task))
    alpha(0.3)
 
end
end