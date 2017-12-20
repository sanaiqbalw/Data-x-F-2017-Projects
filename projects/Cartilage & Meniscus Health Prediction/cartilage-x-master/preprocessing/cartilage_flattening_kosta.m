clear all
close all
clc
LineWidth = 1.5;
addpath('PreProcessing Resources');

%% Setting parameters
% image_file = 'kosta'; % simple masks that I created for debugging
% image_file = 'original'; % the first example that Claudia shared
image_file = 'manual segmentation'; % 2 examples for testing the code on simple splines
% image_file = 'BatchOfData'; % 24 images of real segmentations (separate pieces)
% image_file = 'Cartilage Lesion Example';
% image_file = 'No Cartilage Lesion Example';

splines_type = 'bwboundaries'; % Generating two splines from the segmented cartilage image
% splines_type = 'fake'; % Two circular arcs
% splines_type = 'load from file'; % a good spline, not sure which image does it match to

% image_to_flatten_type = 'image';
image_to_flatten_type = 'segmented';

% show_type = 'imshow';
show_type = 'image';

matching_method = 3; % 1 - Minimum euclidean distances; 2 - Normal vectors; 3 - Laplacian equation

%% Loading images from file
switch image_file
    case 'kosta'
        load('white.mat'); % im_store=Mask
%         load('moon.mat'); % im_store=Mask
%         load('linear.mat'); % im_store=Mask
%         load('parabolic.mat'); % im_store=Mask
        num_start = 1;
        im_store = zeros(512, 512);
        im_store(170:320, 240:400) = 1;
    case 'original'
        load('mri_1.mat'); % im_store
        load('seg_1.mat'); % Mask
        num_start = 20
    case 'manual segmentation'
        man_image_num = 44
%         man_image_num = 45
        load(['D:\Google Drive\Berkeley\Courses\Data X IEOR 290\Project\Cartilage-X\PreProcessing Resources\manualExamplesforKosta\mri_', num2str(man_image_num), '.mat']); % im_store
        load(['D:\Google Drive\Berkeley\Courses\Data X IEOR 290\Project\Cartilage-X\PreProcessing Resources\manualExamplesforKosta\seg_', num2str(man_image_num), '.mat']); % Mask
        num_start = 20
    case 'BatchOfData'
        BatchOfData_image_num = 2005; num_start = 80;
        load(['D:\Berkeley Google Drive\Cartilage-X\BatchOfData\mri\AFACL', num2str(BatchOfData_image_num), '_01.mat']); % im_store
        load(['D:\Berkeley Google Drive\Cartilage-X\BatchOfData\seg\AFACL', num2str(BatchOfData_image_num), '_01.mat']); % pred_con_vol
        Mask = pred_con_vol;
    case 'Cartilage Lesion Example'
        load('D:\Berkeley Google Drive\Cartilage-X\Cartilage Lesion Example\mri_P346_0_unloaded.mat'); % im_store
        load('D:\Berkeley Google Drive\Cartilage-X\Cartilage Lesion Example\seg_P346_0_unloaded.mat'); % pred_con_vol
        Mask = pred_con_vol;
        num_start = 151
    case 'No Cartilage Lesion Example'
        load('D:\Berkeley Google Drive\Cartilage-X\No Cartilage Lesion Example\mri_P458_0_unloaded.mat'); % im_store
        load('D:\Berkeley Google Drive\Cartilage-X\No Cartilage Lesion Example\seg_P458_0_unloaded.mat'); % pred_con_vol
        Mask = pred_con_vol;
        num_start = 151
end
whos

%% Plotting images that have specific segmented value
if num_start == 0
    value_plot = 1;
    number_of_plots = 0; % 0 for plotting all
    plot_flag = 1;
    indices = plot_images_with_specific_segmented_value(im_store, Mask, value_plot, 0, plot_flag, show_type)
end

%% Plotting a series of consecutive images, original vs segmented
fig_size_2images = [0.66 0.5];
number_of_images = 1
% value_plot = 0 % plot all
value_plot = 1 % MFC, cartilage, medial femoral condyle
% value_plot = 2 % LFC, cartilage, lateral femoral condyle
for image_number = num_start + [0 : number_of_images-1]
        figure
        set(gcf,'units','normalized','outerposition',[0.5-fig_size_2images/2 fig_size_2images]);
        subplot(1,2,1);
        show_image(im_store(:, :, image_number), show_type);
        title(['Image number ', num2str(image_number)]);
        subplot(1,2,2);
        if value_plot == 0
            show_image(Mask(:, :, image_number), show_type);
        else
            seg2d = Mask(:, :, image_number);
            seg2d(seg2d ~= value_plot) = 0;
            show_image(seg2d, show_type);
        end
        title(['Segmented number ', num2str(image_number)]);
end

%% Boundary splines generating/loading from file
switch splines_type
    case 'bwboundaries'
        [boundary, boundary_in, boundary_out, num_sharp_edges] = get_spline_boundaries_old(seg2d, 'kosta');
        if num_sharp_edges <= 2
            figure
            set(gcf,'units','normalized','outerposition',[0.5-fig_size_2images/2 fig_size_2images]);
            subplot(1,2,1);
            show_image(im_store(:, :, image_number), show_type);
            hold on;
            plot(boundary(:, 2), boundary(:, 1), 'g-o', 'LineWidth', LineWidth);
            title(['Image number ', num2str(image_number)]);
            subplot(1,2,2);
            show_image(seg2d, show_type);
            hold on;
            plot(boundary(:, 2), boundary(:, 1), 'g-o', 'LineWidth', LineWidth);
            title(['Segmented number ', num2str(image_number)]);
        end
    case 'fake'
        phase_deg_vec = linspace(-90,180,100)';
        radius_in = 100;
        radius_out = 120;
        boundary_in = [256 + radius_in*cos((pi/180)*phase_deg_vec), 256 + radius_in*sin((pi/180)*phase_deg_vec)];
        boundary_out = [256 + radius_out*cos((pi/180)*phase_deg_vec), 256 + radius_out*sin((pi/180)*phase_deg_vec)];
    case 'load from file'
%         load('D:\Google Drive\Berkeley\Courses\Data X IEOR 290\Project\Cartilage-X\PreProcessing Resources\manualExamplesforKosta\ACL037_130222_E318_CUBE_trans_contra_MFC.mat');
%         slice_num = 10
        load('D:\Google Drive\Berkeley\Courses\Data X IEOR 290\Project\Cartilage-X\PreProcessing Resources\manualExamplesforKosta\ACL037_130222_E318_CUBE_trans_contra_LFC.mat');
        slice_num = 21
        dataslices = 'only one';
%         dataslices = 'all';
        switch dataslices
            case 'only one'
                myinfotosave.w3 = 1;
                myinfotosave.dataperslice = {myinfotosave.dataperslice{slice_num}};
                boundary_in = [myinfotosave.dataperslice{1}.xcoordinatesspl{1} myinfotosave.dataperslice{1}.ycoordinatesspl{1}];
                boundary_out = [myinfotosave.dataperslice{1}.xcoordinatesspl{2} myinfotosave.dataperslice{1}.ycoordinatesspl{2}];
            case 'all'
                boundary_in = [myinfotosave.dataperslice{slice_num}.xcoordinatesspl{1} myinfotosave.dataperslice{slice_num}.ycoordinatesspl{1}];
                boundary_out = [myinfotosave.dataperslice{slice_num}.xcoordinatesspl{2} myinfotosave.dataperslice{slice_num}.ycoordinatesspl{2}];
        end
        disp('size(myinfotosave.dataperslice):'); disp(size(myinfotosave.dataperslice));
        disp('myinfotosave.w3:'); disp(myinfotosave.w3);
end

%% Plotting the original and the segmented images with the splines
figure
set(gcf,'units','normalized','outerposition',[0.5-fig_size_2images/2 fig_size_2images]);
subplot(1,2,1);
show_image(im_store(:, :, image_number), show_type);
hold on;
plot(boundary_in(:, 1), boundary_in(:, 2), 'r-o',...
    boundary_out(:, 1), boundary_out(:, 2), 'b-o','LineWidth', LineWidth);
title(['Image number ', num2str(image_number)]);
subplot(1,2,2);
show_image(seg2d, show_type);
hold on;
plot(boundary_in(:, 1), boundary_in(:, 2), 'r-o',...
    boundary_out(:, 1), boundary_out(:, 2), 'b-o','LineWidth', LineWidth);
title(['Segmented number ', num2str(image_number)]);

%% Flattening
switch splines_type
    case {'bwboundaries', 'fake'}
        switch image_to_flatten_type
            case 'image'
                image_to_flatten = im_store(:, :, image_number);
            case 'segmented'
                image_to_flatten = seg2d;
        end
        myinfotosave = get_myinfotosave(image_to_flatten, boundary_in, boundary_out);
    case 'load from file'
        num_slices_to_flatten = length(myinfotosave.dataperslice);
        switch image_to_flatten_type
            case 'image'
                if (size(im_store,3)==1) & (num_slices_to_flatten > 1)
                    im_store = repmat(im_store, [1, 1, num_slices_to_flatten]);
                end
                image_to_flatten = im_store(:, :, 1:num_slices_to_flatten);
            case 'segmented'
                image_to_flatten = seg2d;
        end
end
% disp(myinfotosave)
% noffsets - Since cartilage flattening is predominantly done for texture analysis using gray-level co-occurrence matrices (GLCM), this is a scalar that tells the program the number of offsets
% to be used in GLCM. The purpose of noffsets is for cases where the cartilage is split into multiple sements in a single slice so we can separate the flattened images accordingly.
noffsets = 5;
isboneref = 1; % the reference for flattening, i.e. what will look like a straight line. isboneref=1: bone-cartilage interface, isboneref=0: articular surface
T2Th = []; % a scalar indicating that all Ima values greater than T2Th should be cropped to T2Th. T2Th=[] does not crop Ima.
displayflag = 0; % displayflag=1 Displays partial results, displayflag=0 Displays nothing.
disp(['noffsets=', num2str(noffsets), ', isboneref=', num2str(isboneref),  ', T2Th=', num2str(T2Th)]);
disp(['Running flat_2D_cartilage_warping() with matching_method=', num2str(matching_method),' ...']);
tic_start = tic;
[fIma, fmyinfotosave, MeanT2OrigFlat, StdT2OrigFlat, Thicks] = flat_2D_cartilage_warping(image_to_flatten, myinfotosave, matching_method, noffsets, isboneref, T2Th, displayflag);
elapsed_time = toc(tic_start); disp(['Elapsed toc time = ', num2str(floor(elapsed_time/60)),' min, ', num2str(round(mod(elapsed_time,60))), ' sec']);
disp('Flattened image size:'); disp(size(fIma{1}));

% save('slice20_mm1_bwboundaries.mat', 'fIma', 'fmyinfotosave', 'MeanT2OrigFlat', 'StdT2OrigFlat', 'Thicks');
% load('slice20_mm1_bwboundaries.mat');

% fIma_index = 10;
for fIma_index = 1:length(fIma)
    if ~isempty(fIma{fIma_index})
        disp(['fIma_index = ', num2str(fIma_index)]);
        disp('Replacing all NaNs with zeros');
        fIma{fIma_index}(isnan(fIma{fIma_index})) = 0;
        num_nans_ima = sum(sum(isnan(fIma{fIma_index})));
        fig_size_2images_vert = [0.33 1];
        figure
        set(gcf,'units','normalized','outerposition',[0.5-fig_size_2images_vert/2 fig_size_2images_vert]);
        subplot(2,1,1);
        show_image(fIma{fIma_index}, show_type);
%         show_image(fIma{fIma_index}, 'imshow');
%         title(['Original (', num2str(num_nans_ima) ,' NaNs), noffsets=', num2str(noffsets), ', isboneref=', num2str(isboneref),  ', T2Th=', num2str(T2Th),  ', MM=', num2str(matching_method)]);
        title(['Flattened ', image_to_flatten_type ,', slice ', num2str(fIma_index) , ', size=(', num2str(size(fIma{fIma_index},1)), ', ', num2str(size(fIma{fIma_index},2)),'), MM=', num2str(matching_method)]);
        subplot(2,1,2);
        fboundary_in = [fmyinfotosave.dataperslice{fIma_index}.xcoordinatesspl{1} fmyinfotosave.dataperslice{fIma_index}.ycoordinatesspl{1}];
        fboundary_out = [fmyinfotosave.dataperslice{fIma_index}.xcoordinatesspl{2} fmyinfotosave.dataperslice{fIma_index}.ycoordinatesspl{2}];
        plot(fboundary_in(:, 1), fboundary_in(:, 2), 'r-o',...
            fboundary_out(:, 1), fboundary_out(:, 2), 'b-o','LineWidth', LineWidth);
        title(['Flattened splines, slice ', num2str(fIma_index) , ', length=', num2str(size(fboundary_in,1))]);
        disp('size(fIma):'); disp(size(fIma{fIma_index}));
        disp('size(fboundary_in):'); disp(size(fboundary_in));
        disp('size(fboundary_out):'); disp(size(fboundary_out));
    end
end
