% This demo plots a soft star and then uses ROger Stafford's formula
% to find and mark the locations on the star
% that have high curvature.
% Initialization & clean up stuff.
clc;    % Clear the command window.
close all;  % Close all figures (except those of imtool.)
clear;  % Erase all existing variables.
% workspace;  % Make sure the workspace panel is showing.

format longg;
format compact;
fontSize = 20;

%=====================================================
% First make a shape with sharp turns or cusps.
% Demo macro to draw a rounded star (like a splat).
% Select the inner and outer radius.
outerRadius = 44  % You can change this
innerRadius = 19  % You can change this
% Select the number of lobes around the circle.
numberOfLobes = 8;  % You can change this
period = 2 * pi / numberOfLobes;
meanRadius = (outerRadius + innerRadius)/2
amplitude = (outerRadius - innerRadius)/2
t = (0:.005:1)*2*pi; % Independent parameter.
variableRadius = amplitude * cos(2*pi*t/period) + meanRadius;
subplot(2,2,1);
plot(variableRadius, 'LineWidth', 2);
grid on;
ylim([0 outerRadius]);
title('VariableRadius', 'FontSize', fontSize);
period = 2*pi;  % Need to change this now.
xStar = variableRadius .* cos(2*pi*t/period); 
yStar = variableRadius .* sin(2*pi*t/period); 
subplot(2,2,2);
plot(t, xStar, 'LineWidth', 2);
grid on;
title('x2 vs. t', 'FontSize', fontSize);
subplot(2,2,3);
plot(t, yStar, 'LineWidth', 2);
grid on;
title('y2 vs. t', 'FontSize', fontSize);
subplot(2,2,4);
plot(xStar, yStar,'b.', 'LineWidth', 2) 
title('x2 vs y2', 'FontSize', fontSize);
axis square;
% Maximize window.
set(gcf, 'units','normalized','outerposition',[0 0 1 1]); % Maximize figure.
set(gcf,'name','Image Analysis Demo','numbertitle','off') 
% OK - all of the above code was just to get some demo data
% that we can use to find the high radius of curvature locations on.

%=====================================================
% Now run along the (x2, y2) soft star curve 
% and find the radius of curvature at each location.
numberOfPoints = length(xStar);
curvature = get_curvature(xStar, yStar, 'Roger');
% Plot curvature.
figure;
subplot(2, 1, 1);
plot(curvature, 'b-', 'LineWidth', 2);
grid on;
xlim([1 numberOfPoints]); % Set limits for the x axis.
title('Radius of Curvature', 'FontSize', fontSize);
% Find high curvature points - 
% indexes where the curvature is greater than 0.3
highCurvatureIndexes = find(curvature > 0.3);
% Plot soft star again so we can plot 
% high curvature points over it.
subplot(2, 1, 2);
plot(xStar, yStar,'b', 'LineWidth', 2) 
grid on;
axis square;
% Mark high curvature points on the star.
hold on;
plot(xStar(highCurvatureIndexes), yStar(highCurvatureIndexes), ...
	'rd', 'MarkerSize', 15, 'LineWidth', 2);
title('Soft Star with High Curvature Locations Marked', 'FontSize', fontSize);
% Maximize window.
set(gcf, 'units','normalized','outerposition',[0 0 1 1]); % Maximize figure.
set(gcf,'name','Image Analysis Demo','numbertitle','off');
