clear all
close all
clc
addpath('PreProcessing Resources');

image_file_path = '/tools/projects/kosta/matlab/BatchOfData';
%%
% Testing 3D sphere strel, comparing morphology of resulting structure
%
% Load in an example of a crappy segmentation:

% BatchOfData_image_num = 2006;
for BatchOfData_image_num = [[2005:2009] [2011:2022] [3001:3003] [3005:3008]]
% for BatchOfData_image_num = [[2007:2009] [2011:2022] [3001:3003] [3005:3008]]
    disp(['Processing image ', num2str(BatchOfData_image_num),' ...']);
    load([image_file_path, '/seg/AFACL', num2str(BatchOfData_image_num), '_01.mat']); % im_store
    % load('/Users/talairach/Downloads/flatten/AFACL2005_01.mat')
    %%
    % Segmentation is in pred_con_vol, can only imshow one "slice" at a time,
    % default view is sagittal, but using permute function we can visualize coronal
    % and axial
    %
    % Goal: compare visually (2 subplots per view: sag, cor, ax) and quant (regionprops
    % "SurfaceArea", "Volume" and "Bounding Box")
    %
    % IDEAL: Don't want a change in bounding box, but SA and V should change
    %
    % Test se = sphere of size [1 2 4 6]

    no_edit_sag = double(pred_con_vol==1);
    tba_edit_sag = no_edit_sag;

    no_edit_props = regionprops3(no_edit_sag, {'Volume','BoundingBox','SurfaceArea'});
    VOL(1) = no_edit_props.Volume;
    BB(1,:) = no_edit_props.BoundingBox;
    SA(1) = no_edit_props.SurfaceArea;

    %%
    % Alternatively, can write function which takes in matrix and shows all
    % views

    % r = [1 2 4 6];
    r = [6];
    for j = 1:length(r)
        disp(['r = ', num2str(r(j))]);
        se = strel('sphere',r(j));
    %     se = strel('ball', r(j), r(j), r(j));
        cl_sag = imclose(tba_edit_sag,se);
        cl_sag_logical = logical(cl_sag);
        save([image_file_path, '/seg_imclose3/AFACL', num2str(BatchOfData_image_num), '_01.mat'], 'cl_sag_logical');

        cl_props = regionprops3(cl_sag, {'Volume','BoundingBox','SurfaceArea'});
        VOL(j+1) = cl_props.Volume;
        BB(j+1,:) = cl_props.BoundingBox;
        SA(j+1) = cl_props.SurfaceArea;

    %     no_edit_cor = permute(no_edit_sag,[1 3 2]);
    %     no_edit_ax = permute(no_edit_sag, [2 3 1]);
    %     
    %     cl_cor = permute(cl_sag,[1 3 2]); % cor
    %     cl_ax = permute(cl_sag, [2 3 1]); % ax
    %     
    %     for i = [10]%1:24
    %         figure(2);
    %         subplot(1,2,1)
    %         imshow(no_edit_sag(:,:,70+i),[])
    %         subplot(1,2,2)
    %         imshow(cl_sag(:,:,70+i),[])
    %         title(['SagView r=',num2str(r(j))])
    %         
    %         figure(3);
    %         subplot(1,2,1)
    %         imshow(no_edit_cor(:,:,171+i),[])
    %         subplot(1,2,2)
    %         imshow(cl_cor(:,:,171+i),[])
    %         title(['CorView r=',num2str(r(j))])
    %         
    %         figure(4);
    %         subplot(1,2,1)
    %         imshow(no_edit_ax(:,:,255+i),[])
    %         subplot(1,2,2)
    %         imshow(cl_ax(:,:,255+i),[])
    %         title(['AxView r=',num2str(r(j))])

    %         waitforbuttonpress
    %     end
    end
    %%
    % We collect and compare whole volume stats
    VOL
    BB
    SA
end
