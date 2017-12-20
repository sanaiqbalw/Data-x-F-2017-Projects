%% PCA on 2D linear data distribution
% SLectured by Sujin Jang in ECE662 2014 Spring, Purdue University

clear all;
close all;

% Generate linear 2D data
x(1,:) = randn(1,100);
x(2,:) = randn(1,100)*3;
[p(1,:),p(2,:)] = cart2pol(x(1,:),x(2,:));
p(1,:) = p(1,:)-pi/3;
[x(1,:),x(2,:)] = pol2cart(p(1,:),p(2,:));

scatter(x(1,:),x(2,:), 'ok');
title('Raw 2D data distribution', 'FontSize',15);
xlabel('x_1','FontSize',15); ylabel('x_2', 'FontSize',15);
axis equal;

%%
% SVD
[U,S,V] = svd(cov(x'));
pc = V;

% Plot principal component direction
hold on;
line([-10*pc(1,1),10*pc(1,1)], [-10*pc(2,1),10*pc(2,1)], 'Color', [1 0 0]);
hold on;
line([-5*pc(1,2),5*pc(1,2)], [-5*pc(2,2),5*pc(2,2)], 'Color', [0 0 1]);
legend('2-D data','1st eigenvector dir.','2nd eigenvector dir');

%%
% Rotate data matrix using principal axes
x_rot = pc'*x;

% Plot rotated data
figure;
scatter(x_rot(1,:), x_rot(2,:), 'ok');
title('Rotated 2D data distribution', 'FontSize',15);
xlabel('x''_1','FontSize',15); ylabel('x''_2', 'FontSize',15);
axis equal;

% SVD for rotated x matrix
[U,S,V] = svd(cov(x_rot'));
pc_rot = V;
% [pc2, latent, explained] = pcacov(cov(y'));

% Plot rotated principal directions
hold on;
line([-10*pc_rot(1,1),10*pc_rot(1,1)], [-10*pc_rot(2,1), 10*pc_rot(2,1)], 'Color', [1 0 0]);
hold on;
line([-5*pc_rot(1,2), 5*pc_rot(1,2)], [-5*pc_rot(2,2), 5*pc_rot(2,2)], 'Color', [0 0 1]);
legend('2-D data','1st eigenvector dir.','2nd eigenvector dir');
  
%% Reproject rotated data into original space
trY = x_rot;  
trY(2,:) = 0;
trX = (trY'*inv(pc))';
figure;
scatter(x(1,:),x(2,:),'ok');
hold on;
scatter(trX(1,:),trX(2,:),'or');
hold on;
error1 = 0;
for i=1:size(x,2)
  line([trX(1,i),x(1,i)], [trX(2,i),x(2,i)]);
  error1 = error1 + norm([trX(1,i)-x(1,i); trX(2,i)-x(2,i)],2);
  hold on;
end
title('Projection on the primary eigenvector', 'FontSize',15);
xlabel('x_1','FontSize',15); ylabel('x_2', 'FontSize',15);
legend('Original data', 'Projected data', 'Projection error');
axis equal;

x_rot(1,:) = 0;
trX = (x_rot'*inv(pc))';
figure;
scatter(x(1,:),x(2,:),'ok');
hold on;
scatter(trX(1,:),trX(2,:),'or');
hold on;
error2 = 0;
for i=1:size(x,2)
  line([trX(1,i),x(1,i)], [trX(2,i),x(2,i)]);
  error2 = error2 + norm([trX(1,i)-x(1,i); trX(2,i)-x(2,i)],2);
  hold on;
end
title('Projection on the secondary eigenvector', 'FontSize',15);
xlabel('x_1','FontSize',15); ylabel('x_2', 'FontSize',15);
legend('Original data', 'Projected data', 'Projection error');
axis equal;
error1 
error2
