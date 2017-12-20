clear all
close all
clc
LineWidth = 1.5;
addpath('PreProcessing Resources');

image_file_path = 'D:\Berkeley Google Drive\Cartilage-X\BatchOfData';
image_value_to_select = 1; % MFC, cartilage, medial femoral condyle

%% Choose the script type
% script_type = 'run'; % runs the flattening and saves the result to a file
script_type = 'plot'; % loads a saved flattening file and plots the result

%% Choose the function for plotting the images
% show_type = 'imshow';
% show_type = 'image';
show_type = 'imagesc';

%% Run through the images in the image_file_path folder
start_time = datetime('now')
% for BatchOfData_image_num = [[2015:2022] [3001:3003] [3005:3008]]
% for BatchOfData_image_num = [[2005:2009] [2011:2022] [3001:3003] [3005:3008]]
for BatchOfData_image_num = [3002]
    disp(['Processing image ', num2str(BatchOfData_image_num),' ...']);
    load([image_file_path, '\seg\AFACL', num2str(BatchOfData_image_num), '_01.mat']); % pred_con_vol
    Mask = pred_con_vol;

    % getting the z indices of the masks with the right value
    plot_flag = 0;
    disp(['Image ', num2str(BatchOfData_image_num),' has ', num2str(size(Mask,3)), ' slices']);
    disp(['Indices of slices that have segmented values of ', num2str(image_value_to_select), ':']);
    z_indices = plot_images_with_specific_segmented_value(Mask, Mask, image_value_to_select, 0, plot_flag, show_type)
%     z_indices = [188];

    R_disk_vec = zeros(size(z_indices)); % the size of strel disk used in imdilate() if imclose() was not sufficient
    non_zero_indices = 1:length(z_indices);
    max_figures_to_plot = 15; % plotting too many figures fills the memory and Matlab gets stuck...
    if (strcmp(script_type, 'plot') == 1) && (length(z_indices) > max_figures_to_plot)
        plot_chunk = 1;
        non_zero_indices = [(plot_chunk-1)*max_figures_to_plot+1 : min([plot_chunk*max_figures_to_plot, length(z_indices)])];
    end
    for non_zero_ind = non_zero_indices
        z_index = z_indices(non_zero_ind);
        switch script_type
            case 'run'
                disp(['Processing z index ', num2str(z_index),' ...']);
                seg2d = Mask(:, :, z_index);
                seg2d(seg2d ~= image_value_to_select) = 0;
                seg2d = bwmorph(seg2d, 'clean'); % remove single pixels
                seg2d_before_dilate = imfill(seg2d, 'holes'); % fill the holes
                seg2d_final = seg2d_before_dilate;
                [B,L] = bwboundaries(seg2d_final, 'noholes'); % find the spline boundaries
                R_disk = 0;
                if length(B) > 1 % there is more than one piece, try imclose()
                    seg2d_before_dilate = imclose(seg2d_before_dilate, strel('disk', 50, 6));  % combine the neighbors
                    seg2d_final = seg2d_before_dilate;
                    [B,L] = bwboundaries(seg2d_final, 'noholes'); % find the spline boundaries          
                    while length(B) > 1 % dilate if there is still more than one piece
                        R_disk = R_disk + 1;
                        seg2d_final = imdilate(seg2d_final, strel('disk', R_disk)); 
                        [B,L] = bwboundaries(seg2d_final, 'noholes');
                    end
                end

                %% Finding the spline boundaries
                if R_disk < 6
                    method = 'endpoints';
                else
                    method = 'polyfit';
                end
                [flag_good, boundary, boundary_in, boundary_out] = get_spline_boundaries2(seg2d_final, method);

                if flag_good == 1 % the segmentation is large enough, do the flattening
                    %% Flattening the segmentation
                    [seg_flattened, seg_boundary_flattened] = flat_2D_cartilage_warping_claudia(double(seg2d_final), boundary_in, boundary_out, 0);

                    %% Flattening the actual image
                    load([image_file_path, '\mri\AFACL', num2str(BatchOfData_image_num), '_01.mat']); % im_store
                    im_2D = im_store(:,:,z_index);
                    [im_flattened, im_boundary_flattened] = flat_2D_cartilage_warping_claudia(im_2D, boundary_in, boundary_out, 0);
                    im_flattened_clipped = im_flattened;
                    im_flattened_clipped(im_flattened_clipped > 1) = 1;
                    im_flattened_clipped(im_flattened_clipped < 0) = 0;

                    %% Saving the results
                    save([image_file_path, '\flatten\AFACL', num2str(BatchOfData_image_num), '_01_slice_', num2str(z_index), '.mat'], ...
                        'BatchOfData_image_num', 'z_index', 'flag_good', 'boundary', 'boundary_in', 'boundary_out', 'R_disk', ...
                        'seg2d', 'seg2d_before_dilate', 'seg2d_final', 'seg_flattened', 'seg_boundary_flattened', ...
                        'im_2D', 'im_flattened_clipped', 'im_boundary_flattened', 'show_type');
                else % flag_good = 0, the segmentation is too small, do nothing
                    load([image_file_path, '\mri\AFACL', num2str(BatchOfData_image_num), '_01.mat']); % im_store
                    im_2D = im_store(:,:,z_index);
                    save([image_file_path, '\flatten\AFACL', num2str(BatchOfData_image_num), '_01_slice_', num2str(z_index), '.mat'], ...
                        'BatchOfData_image_num', 'z_index', 'flag_good', 'boundary', 'R_disk', 'seg2d', 'seg2d_before_dilate', 'seg2d_final', 'im_2D', 'show_type');
                end

            case 'plot'
                %% Load and plot the results
                load([image_file_path, '\flatten\AFACL', num2str(BatchOfData_image_num), '_01_slice_', num2str(z_index), '.mat']);
                if flag_good == 1
                    plot_flattening_results(BatchOfData_image_num, z_index, boundary, boundary_in, boundary_out, R_disk, ...
                        seg2d, seg2d_before_dilate, seg2d_final, seg_flattened, seg_boundary_flattened, ...
                        im_2D, im_flattened_clipped, im_boundary_flattened, show_type);
                else % flag_good = 0
                    plot_non_flattened_results(BatchOfData_image_num, z_index, boundary, R_disk, seg2d, seg2d_before_dilate, seg2d_final, im_2D, show_type);
                end
        end
        R_disk_vec(non_zero_ind) = R_disk;
    end
    %% Plotting Rdisk
    if strcmp(script_type, 'plot') == 1
        figure
        stem(z_indices, R_disk_vec, '-o', 'LineWidth', LineWidth);
        xlabel('z index'); ylabel('R_{disk}');
        grid on;
    end
end
end_time = datetime('now')
elapsed_time = etime(datevec(end_time), datevec(start_time));
disp(['Elapsed time = ', num2str(floor(elapsed_time/60)),' min, ', num2str(round(mod(elapsed_time,60))), ' sec']);
