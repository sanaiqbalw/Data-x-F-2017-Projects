function testingJoin
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
%I = im2bw(imread('http://i.imgur.com/Bl7zhcn.jpg'));
A = load('/Users/talairach/Downloads/seg/AFACL2020_01.mat');
B = A.pred_con_vol;
B(B~=1)=0;
%I = imread('/Users/talairach/Desktop/cart_shape.jpg');
% your original image
%I=im2bw(I);
figure,imshow(B(:,:,168))
%I= I(5:end-4,5:end-4);
% im0 = B(:,:,165)
se = strel('disk',6);
im1a = imclose(B(:,:,168),se);
figure,imshow(im1a,[])
im1 = bwmorph(im1a,'thin',Inf);
[x,y] = find(bwmorph(im1,'endpoints'));
for iter = 1:numel(x)-1
im1=linept2(im1, x(iter), y(iter), x(iter+1), y(iter+1));
end
im2=imfill(im1,'holes');
figure,imshow(im2);
% BW = edge(im2);
% figure,imshow(BW);
se = strel('disk', 3);
im3 = imdilate(im2,se);
figure,imshow(im3); hold on;
im4 = bwmorph(im3,'thin',Inf);
figure, imshow(im4)
res = or(im1a,im4);
imshow(imfill(res,'holes'))
K = imfill(res,'holes');
K2 = bwmorph(K,'fill',10);
imshow(K2)
im5 = imdilate(im4,se);
figure,imshow(im5); hold on;

im1 = bwmorph(im1a,'thin',Inf);
% [x1,y1] = find(bwmorph(im4,'endpoints'));
% scatter(y1,x1)

% 
% seg2d_fill = imfill(B(:,:,165), 'holes');
% close_seg2d_fill_dilated = imdilate(close_seg2d_fill, strel('disk', 1));
% imshow(close_seg2d_fill_dilated,[])
% fig_size = [0.66 1];
% plot_num_x = 2; plot_num_y = 2;
% axis_vec = [0 511 0 511];


end

