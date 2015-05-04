%
% Investigation of rolling shutter effect on 2D feature positions
% Tested with Octave 3.6.4
%

clf
close all
clc

load rolling_shutter_features_out.mat    % data
dataNormX = data(:,:,1) - mean(data(:,:,1));
dataNormY = data(:,:,2) - mean(data(:,:,2));

FontSize = 20;

% Plot all feature points
figure(1)
title('All (2D) feature points')
hold on
for i=1:size(data,2)
    plot(data(:,i,1), data(:,i,2), 'o-')
end
set(gca, 'FontSize', 12)
set(gca, 'YDir', 'reverse')
axis equal tight
hold off
xlabel('X coordinate in image [px]')
ylabel('Y coordinate in image [px]')
saveas(gcf, ['figures/all_features.pdf'])

% Plot features' mean deviation over time for X and Y coordinates
figure(2)
plot(dataNormX, 'o-')
ylim([-40, 40])
title('All X-coordinates of feature points vs time')
set(gca, 'FontSize', FontSize)
xlabel('framenumber in sequence')
ylabel('X deviation from mean [px]')
saveas(gcf, ['figures/all_features_X.pdf'])
figure(3)
plot(dataNormY, 'o-')
ylim([-40, 40])
title('All Y-coordinates of feature points vs time')
set(gca, 'FontSize', FontSize)
xlabel('framenumber in sequence')
ylabel('Y deviation from mean [px]')
saveas(gcf, ['figures/all_features_Y.pdf'])

% Classify data into rolling shutter error and bad tracks
dataANormX = abs(dataNormX);
dataANormY = abs(dataNormY);
class0    =                                        find(all(dataANormX == 0));
classHalf = intersect(find(any(dataANormX > 0)),   find(all(dataANormX <= 0.5)));
class1    = intersect(find(any(dataANormX > 0.5)), find(all(dataANormX <= 1)));
class3    = intersect(find(any(dataANormX > 1)),   find(all(dataANormX <= 3)));
class3pX  =                                        find(any(dataANormX >  3));    % bad tracks
class3pY  =                                        find(any(dataANormY >  3));    % bad tracks
class3p   = intersect(class3pX, class3pY);    % bad tracking error for both coordinates

disp (['size(data)      = ', num2str(size(data,2)),      '    std = ', num2str(std(dataNormX(:,:)(:)))])
disp (['size(class0)    = ', num2str(length(class0)),    '    std = ', num2str(std(dataNormX(:,class0)(:)))])
disp (['size(classHalf) = ', num2str(length(classHalf)), '    std = ', num2str(std(dataNormX(:,classHalf)(:)))])
disp (['size(class1)    = ', num2str(length(class1)),    '    std = ', num2str(std(dataNormX(:,class1)(:)))])
disp (['size(class3)    = ', num2str(length(class3)),    '    std = ', num2str(std(dataNormX(:,class3)(:)))])
disp (['size(class3p)   = ', num2str(length(class3p)),   '    std = ', num2str(std(dataNormX(:,class3p)(:)))])

% Visualize class 0.5
figure(4)
plot([dataNormX(:,classHalf(1)), dataNormY(:,classHalf(1))], 'o-')
title('Class 0.5 (only first feature shown)    X and Y vs time')
figure(5)
hist(dataNormX(:,classHalf(1)))
title('Class 0.5    Histogram X')
figure(6)
hist(dataNormY(:,classHalf(1)))
title('Class 0.5    Histogram Y')

% Visualize class 1
figure(7)
plot([dataNormX(:,class1(1)), dataNormY(:,class1(1))], 'o-')
ylim([-1, 1.5])
title('Class 1 (only first feature shown)    X and Y vs time')
set(gca, 'FontSize', FontSize)
xlabel('framenumber in sequence')
ylabel('deviation from mean [px]')
set(legend('X coordinate', 'Y coordinate'), 'FontSize', FontSize)
saveas(gcf, ['figures/class1_deviation.pdf'])
figure(8)
hist(dataNormX(:,class1(1)))
title('Class 1    Histogram X')
figure(9)
hist(dataNormY(:,class1(1)))
title('Class 1    Histogram Y')

% Visualize class 3
figure(10)
plot([dataNormX(:,class3(1)), dataNormY(:,class3(1))], 'o-')
ylim([-1, 1.5])
title('Class 3 (only first feature shown)    X and Y vs time')
set(gca, 'FontSize', FontSize)
xlabel('framenumber in sequence')
ylabel('deviation from mean [px]')
set(legend('X coordinate', 'Y coordinate'), 'FontSize', FontSize)
saveas(gcf, ['figures/class3_deviation.pdf'])
figure(11)
hist(dataNormX(:,class3(1)))
title('Class 3    Histogram X')
figure(12)
hist(dataNormY(:,class3(1)))
title('Class 3    Histogram Y')

% Visualize class 3+
figure(13)
plot(data(:,class3p(1),1), data(:,class3p(1),2), 'o-')
title('Class 3+ (only first feature shown)    XY vs time')
NumTicks = 3; L = get(gca, 'XLim'); set(gca, 'XTick', linspace(L(1),L(2),NumTicks))
set(gca, 'FontSize', FontSize)
set(gca, 'YDir', 'reverse')
axis equal tight
xlabel('X coordinate in image [px]')
ylabel('Y coordinate in image [px]')
saveas(gcf, ['figures/class3plus_drift.pdf'])
figure(14)
hist(dataNormX(:,class3p(1)))
title('Class 3+    Histogram X')
figure(15)
hist(dataNormY(:,class3p(1)))
title('Class 3+    Histogram Y')

% Visualize outliers caused by bad tracks    (95% of the data seems to have a std < 1.0)
dataNormRadius = sqrt(dataNormX.^2 + dataNormY.^2);
sortedStdRadius = sort(std(dataNormRadius));
figure(16)
semilogy(linspace(0, 100, size(data,2)), sortedStdRadius)
title('Error of a feature over time vs percentile')
set(gca, 'FontSize', 17)
xlabel('percentile in the sorted list of all features [%]')
ylabel('sigma of error over time of a feature [px]')
saveas(gcf, ['figures/radius_error_distribution.pdf'])

L_1sigma = round(.6827 * length(sortedStdRadius));
L_2sigma = round(.9545 * length(sortedStdRadius));
L_3sigma = round(.9973 * length(sortedStdRadius));
std_1sigma = sqrt(mean(sortedStdRadius(1:L_1sigma).^2));
std_2sigma = sqrt(mean(sortedStdRadius(1:L_2sigma).^2));
std_3sigma = sqrt(mean(sortedStdRadius(1:L_3sigma).^2));
disp (['std of Euclidean error of feature over time, within 1 sigma over all features = ', num2str(std_1sigma)])
disp (['std of Euclidean error of feature over time, within 2 sigma over all features = ', num2str(std_2sigma)])
disp (['std of Euclidean error of feature over time, within 3 sigma over all features = ', num2str(std_3sigma)])
