clear all
close all
clc

% filename = 'white';
% filename = 'moon';
% Mask = imread(strcat(filename,'.bmp'));
% Mask = double(Mask);
% Mask(Mask == 15) = 1;
% whos
% 
% Mask = imrotate(Mask, 180);
% 
% figure
% imshow(Mask);
% 
% im_store = Mask;
% save(strcat(filename,'.mat'), 'im_store', 'Mask');

%% Creating manual gradual mask
[x,y] = meshgrid(0:511, 0:511);
fill_type = 'linear';
% fill_type = 'parabolic';
switch fill_type
    case 'linear'
%         Mask = x + y;
        Mask = 511 - x + y;
    case 'parabolic'
        Mask = x.^2 + y.^2;
end
Mask = Mask/max(max(Mask));

figure
image(64*Mask);
% image(Mask, 'CDataMapping', 'scaled');
colorbar;

im_store = Mask;
save(strcat(fill_type,'.mat'), 'im_store', 'Mask');
