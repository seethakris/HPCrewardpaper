%%Finds slope of data x position for given tasks
%Chad Heer; Sheffield lab

%INPUTS
%data = binned data (lap x bin)
%position = time or position of each bin
%tasks = vector array of task identity

%OUTPUTS
%lap_slope = slope of line fit to binned data for each lap
%line_r2 = R2 of line fit to data
%coeffs = coefficient of exponential curve fit to data
%exp_r2 = R2 of exponential curve fit to data

function[lap_slope, line_r2, coeffs, exp_r2] = find_slope(data, position, tasks)
%find exponential fit to the first lap
template = nanmean(data{1})/max(nanmean(data{1}));
[min_val, min_idx] = min(template);

[temp_fit, temp_gof] = fit(position(min_idx:34)', template(min_idx:34)','exp1')
coeff = coeffvalues(temp_fit);

for task = 1: length(tasks)
    %normalize data
    norm_data{task} = data{task}/max(nanmean(data{1}));
    
    for lap = 1: size(norm_data{task},1)
        %find the maximum value near end of track
        [y2, x2] = max(norm_data{task}(lap,end-14:end));
        
        if task == 1
            %find the minimum value near the beggining of track 
            [y1(lap), xtask1(lap)] = min(norm_data{task}(lap,1:end-15));
            %find linear fit to data between minimum and maximum values
            lm = fitlm(position(xtask1(lap):25+x2),norm_data{task}(lap,xtask1(lap):25+x2));
        else
        
            [y1(lap), x1(lap)] = min(norm_data{task}(lap,min(xtask1):max(xtask1)));
            lm = fitlm(position(x1(lap)+min(xtask1)-1:25+x2),norm_data{task}(lap,x1(lap)+min(xtask1)-1:25+x2));
        end
        
%         lap_slope(task,lap) = (y2 - y1)/(25+x2 - x1);
        
        
        %pull out lap_slope and line_r2
        lap_slope(task,lap) = lm.Coefficients.Estimate(2);
        
        line_r2(task,lap) = lm.Rsquared.Ordinary;
        
        %fit template exponential equation to laps binned data
        fit_eq = [num2str(coeff(1)) ' * a * exp(x * ' num2str(coeff(2)) ')'];
        
        [exp_fit, gof] = fit(position(min_idx:34)',norm_data{task}(lap,min_idx:34)',fit_eq, 'Start', position(min_idx));
        
        coeffs(task,lap) = coeffvalues(exp_fit);
        
        exp_r2(task,lap) = gof.rsquare;
        
%         figure;
%         hold on
%         plot(lm)
%         plot(position, norm_data{task}(lap,:))
    end
    
end

lap_slope(lap_slope == 0) = NaN;
line_r2(line_r2 == 0) = NaN;
exp_r2(lap_slope == 0) = NaN;
coeffs(coeffs == 0) = NaN;
lap_slope = lap_slope/nanmean(lap_slope(1,:));


end

