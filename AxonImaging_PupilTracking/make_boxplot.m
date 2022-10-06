%% Makes boxplot of two dimensional data, runs a ttest and corrects for multiple comparisons
%Chad Heer; Sheffield lab

function[] = make_boxplot(data)
%make box plot 
figure;
boxplot(data)
hold on
plot(data','-o')

%run paired t-tests
for i = 1:size(data,2)-1
    for j = i+1:size(data,2)
        [h(i,j),p(i,j)] = ttest(data(:,i),data(:,j), 'Alpha', 0.05);
    end
end

%correct for multiple comparisons using bonferroni correction
[p_anova,t,stats] = anova1(data)
multcompare(stats,'CType', 'bonferroni')


end