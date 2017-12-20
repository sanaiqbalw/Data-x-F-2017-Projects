clear all
close all
clc
LineWidth = 1.5;
addpath('PreProcessing Resources');
image_file_path = 'D:\Berkeley Google Drive\Cartilage-X\BatchOfData';

%% Choose the script type
% script_type = 'run_boundaries'; % find the boundaries, drop the small segmentations, pick neighbor segmentations if needed and saves the result to a file
% script_type = 'plot_boundaries';
% script_type = 'run_flattening'; % runs the flattening and saves the result to a file
script_type = 'plot_flattening'; % loads a saved flattening file and plots the result

%% Choose the function for plotting the images
% show_type = 'imshow';
% show_type = 'image';
show_type = 'imagesc';

%% Run through the images in the image_file_path folder
start_time = datetime('now')
% for BatchOfData_image_num = [[2015:2022] [3001:3003] [3005:3008]]
% for BatchOfData_image_num = [[2005:2009] [2011:2022] [3001:3003] [3005:3008]]
for BatchOfData_image_num = [2005]
    disp(['Processing image ', num2str(BatchOfData_image_num),' ...']);
%     load([image_file_path, '\seg\AFACL', num2str(BatchOfData_image_num), '_01.mat']); % pred_con_vol
%     Mask = pred_con_vol;
    load([image_file_path, '\seg_imclose3\AFACL', num2str(BatchOfData_image_num), '_01.mat']); % cl_sag_logical
    Mask = double(cl_sag_logical);

    switch script_type
    %% Find the boundaries
        case 'run_boundaries'
            image_value_to_select = 1; % MFC, cartilage, medial femoral condyle
            % getting the z indices of the masks with the right value
            plot_flag = 0;
            disp(['Image ', num2str(BatchOfData_image_num),' has ', num2str(size(Mask,3)), ' slices']);
            disp(['Indices of slices that have segmented values of ', num2str(image_value_to_select), ':']);
% %             z_indices = plot_images_with_specific_segmented_value(Mask, Mask, image_value_to_select, 0, plot_flag, show_type)
%             z_indices = [84];
        %     if BatchOfData_image_num == 2008
        %         z_indices = [43:83, 85:90]
        %     end

            R_disk_vec = zeros(size(z_indices)); flag_good_vec = zeros(size(z_indices));
            boundary_cell_arr = cell(size(z_indices)); boundary_cell_arr_in = cell(size(z_indices)); boundary_cell_arr_out = cell(size(z_indices));
            seg = zeros(size(Mask)); seg_before_dilate = zeros(size(Mask)); seg_final = zeros(size(Mask));
            for ind = 1:length(z_indices)
                z_index = z_indices(ind);
%                 disp(['Finding boundaries for z index ', num2str(z_index),' ...']);
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
                if R_disk < 6
                    method = 'endpoints';
                else
                    method = 'polyfit';
                end
                [flag_good, boundary, boundary_in, boundary_out] = get_spline_boundaries2(seg2d_final, method);
                flag_good_vec(ind) = flag_good;
                boundary_cell_arr{ind} = boundary;
                boundary_cell_arr_in{ind} = boundary_in;
                boundary_cell_arr_out{ind} = boundary_out;
                R_disk_vec(ind) = R_disk;
                seg(:,:,z_index) = seg2d;
                seg_before_dilate(:,:,z_index) = seg2d_before_dilate;
                seg_final(:,:,z_index) = seg2d_final;
            end
            %% Plotting Rdisk and flag_good
%             fig_size = [0.33 1];
%             figure
%             set(gcf,'units','normalized','outerposition',[0.5-fig_size/2 fig_size]);
%             subplot(2,1,1);
%             stem(z_indices, R_disk_vec, '-o', 'LineWidth', LineWidth);
%             xlabel('z index'); ylabel('R_{disk}');
%             title(['Image number ', num2str(BatchOfData_image_num)]);
%             grid on;
%             subplot(2,1,2);
%             stem(z_indices, flag_good_vec, '-o', 'LineWidth', LineWidth);
%             xlabel('z index'); ylabel('flag good');
%             grid on;
            %% Keeping only flag_good=1 slices
            bad_z_indices = find(flag_good_vec == 0);
            if ~isempty(bad_z_indices)
                seg(:,:,z_indices(bad_z_indices)) = 0;
                seg_before_dilate(:,:,z_indices(bad_z_indices)) = 0;
                seg_final(:,:,z_indices(bad_z_indices)) = 0;
            end
            good_z_indices = find(flag_good_vec == 1);
            z_indices = z_indices(good_z_indices);
            flag_good_vec = flag_good_vec(good_z_indices);
            boundary_cell_arr = boundary_cell_arr(good_z_indices);
            boundary_cell_arr_in = boundary_cell_arr_in(good_z_indices);
            boundary_cell_arr_out = boundary_cell_arr_out(good_z_indices);
            R_disk_vec = R_disk_vec(good_z_indices);
            %% For Rdisk>0 take the closest Rdisk=0 boundary
            z_indices_zero_Rdisk = z_indices;
            for ind = 1:length(z_indices)
                if R_disk_vec(ind) > 0
                    flag_found = 0;
                    delta = 1;
                    while flag_found == 0
                        ind_new = ind + delta;
                        if (ind_new > 0) && (ind_new < length(z_indices)+1)
                            if R_disk_vec(ind_new) == 0
                                z_indices_zero_Rdisk(ind) = z_indices(ind_new);
                                flag_found = 1;
                            end
                        end
                        if delta > 0
                           delta = -delta;
                        else
                            delta = -delta+1;
                        end
                    end
                    boundary_cell_arr(ind) = boundary_cell_arr(ind_new);
                    boundary_cell_arr_in(ind) = boundary_cell_arr_in(ind_new);
                    boundary_cell_arr_out(ind) = boundary_cell_arr_out(ind_new);
                end
            end
            %% Saving to file
            save([image_file_path, '\flatten_boundaries\AFACL', num2str(BatchOfData_image_num), '_01.mat'], ...
                'BatchOfData_image_num', 'z_indices', 'z_indices_zero_Rdisk', 'flag_good_vec', 'R_disk_vec', ...
                'boundary_cell_arr', 'boundary_cell_arr_in', 'boundary_cell_arr_out', ...
                'seg', 'seg_before_dilate', 'seg_final');
            
            
        case 'plot_boundaries'
            load([image_file_path, '\flatten_boundaries\AFACL', num2str(BatchOfData_image_num), '_01.mat']);
            %% Plotting Rdisk, slices mapping for Rdisk>0 and flag_good
            fig_size = [0.66 1];
            figure
            set(gcf,'units','normalized','outerposition',[0.5-fig_size/2 fig_size]);
            subplot(2,2,1);
            stem(z_indices, R_disk_vec, '-o', 'LineWidth', LineWidth);
            grid on; xlabel('z index'); ylabel('R_{disk}');
            title(['Image number ', num2str(BatchOfData_image_num)]);
            subplot(2,2,2);
            stem(z_indices, flag_good_vec, '-o', 'LineWidth', LineWidth);
            grid on; xlabel('z index'); ylabel('flag good');
            subplot(2,2,3);
            plot(z_indices, z_indices, 'b', z_indices, z_indices_zero_Rdisk, 'ro-', 'LineWidth', LineWidth);
            grid on; xlabel('z index'); ylabel('z index for R_{disk}=0');
%             subplot(2,2,4);
%             plot(boundary_max(:,2), boundary_max(:,1), 'bo', ...
%                 boundary_min(:,2), boundary_min(:,1), 'ro', 'LineWidth', LineWidth);
%             grid on; legend('max', 'min', 'Location', 'NorthWest');
            
            
        case 'run_flattening'
            load([image_file_path, '\flatten_boundaries\AFACL', num2str(BatchOfData_image_num), '_01.mat']);
            z_indices
            seg_flattened_cell = cell(size(z_indices));
            seg_boundary_flattened = cell(size(z_indices));
            im_flattened_cell = cell(size(z_indices));
            im_boundary_flattened = cell(size(z_indices));
            for ind = 1:length(z_indices)
                z_index = z_indices(ind);
                disp(['Flattening z index ', num2str(z_index),' ...']);
                %% Flattening the segmentation
                [seg2d_flattened, seg2d_boundary_flattened] = flat_2D_cartilage_warping_claudia(double(seg_final(:,:,z_index)), boundary_cell_arr_in{ind}, boundary_cell_arr_out{ind}, 0);
                seg_flattened_cell{ind} = seg2d_flattened;
                seg_boundary_flattened{ind} = seg2d_boundary_flattened;
                
                %% Flattening the actual image
                load([image_file_path, '\mri\AFACL', num2str(BatchOfData_image_num), '_01.mat']); % im_store
                [im2d_flattened, im2d_boundary_flattened] = flat_2D_cartilage_warping_claudia(im_store(:,:,z_index), boundary_cell_arr_in{ind}, boundary_cell_arr_out{ind}, 0);

                im2d_flattened_clipped = im2d_flattened;
                im2d_flattened_clipped(im2d_flattened_clipped > 1) = 1;
                im2d_flattened_clipped(im2d_flattened_clipped < 0) = 0;
                im_flattened_cell{ind} = im2d_flattened_clipped;
                im_boundary_flattened{ind} = im2d_boundary_flattened;
            end
            %% Cropping the flattened images (cell arrays take huge space on disc...)
            boundary_flattened_max = zeros(length(z_indices), 2); boundary_flattened_min = zeros(length(z_indices), 2);
            flattened_size = zeros(length(z_indices), 2);
            for ind = 1:length(z_indices)
                boundary_flattened_max(ind, :) = max(seg_boundary_flattened{ind});
                boundary_flattened_min(ind, :) = min(seg_boundary_flattened{ind});
                flattened_size(ind, :) = [size(seg_flattened_cell{ind},1), size(seg_flattened_cell{ind},2)];
            end
            boundary_flattened_size = boundary_flattened_max - boundary_flattened_min;
            boundary_flattened_center = 0.5*(boundary_flattened_max + boundary_flattened_min)
            boundary_d_side = 30;
            crop_size = ceil(max(boundary_flattened_size) + 2*boundary_d_side)
            seg_flattened = zeros(crop_size(2), crop_size(1), length(z_indices));
            im_flattened = zeros(crop_size(2), crop_size(1), length(z_indices));
%             ind_plot = 14
            for ind = 1:length(z_indices)
                if (flattened_size(ind, 1) > crop_size(1))  % there are some weird case where there were some very small flattened sizes on one side
                    start2 = round(boundary_flattened_center(ind,2)) - floor(crop_size(2)/2);
                    start1 = round(boundary_flattened_center(ind,1)) - floor(crop_size(1)/2);
                    indices_flatten2 = start2 + [0 : crop_size(2)-1];
                    indices_flatten1 = start1 + [0 : crop_size(1)-1];
                    if (start1 + crop_size(1) > size(seg_flattened_cell{ind},2))
                        start1 = size(seg_flattened_cell{ind},2) - crop_size(1) - 1;
                        indices_flatten1 = start1 + [0 : crop_size(1)-1];
                    end
                    if (start2 + crop_size(2) > size(seg_flattened_cell{ind},1))
                        start2 = size(seg_flattened_cell{ind},1) - crop_size(2) - 1;
                        indices_flatten2 = start2 + [0 : crop_size(2)-1];
                    end
                    if (start1 > 0) && (start2 > 0)
                        seg_flattened(:,:,ind) = seg_flattened_cell{ind}(indices_flatten2, indices_flatten1);
                        im_flattened(:,:,ind) = im_flattened_cell{ind}(indices_flatten2, indices_flatten1);
                        seg_boundary_flattened{ind} = seg_boundary_flattened{ind} - ones(size(seg_boundary_flattened{ind},1),1)*[start1 start2];
                        im_boundary_flattened{ind} = im_boundary_flattened{ind} - ones(size(seg_boundary_flattened{ind},1),1)*[start1 start2];
                    end
                end
            end
%             figure
%             show_image(seg_flattened_cell{ind_plot}, show_type);
%             figure
%             show_image(seg_flattened(:,:,ind_plot), show_type);
%             hold on;
%             plot(seg_boundary_flattened{ind_plot}(:, 1), seg_boundary_flattened{ind_plot}(:, 2), 'g-o', 'LineWidth', LineWidth);
%             figure
%             show_image(seg_flattened(:,:,2), show_type);
%             hold on;
%             plot(seg_boundary_flattened{2}(:, 1), seg_boundary_flattened{2}(:, 2), 'g-o', 'LineWidth', LineWidth);
           
            %% Saving the results
            save([image_file_path, '\flatten\AFACL', num2str(BatchOfData_image_num), '_01.mat'], ...
                'seg_flattened', 'seg_boundary_flattened', 'im_flattened', 'im_boundary_flattened');
          
            
            case 'plot_flattening'
                %% Load and plot the results
                load([image_file_path, '\flatten\AFACL', num2str(BatchOfData_image_num), '_01.mat']);
                load([image_file_path, '\flatten_boundaries\AFACL', num2str(BatchOfData_image_num), '_01.mat']);
                load([image_file_path, '\mri\AFACL', num2str(BatchOfData_image_num), '_01.mat']); % im_store
                
                max_figures_to_plot = 15; % plotting too many figures fills the memory and Matlab gets stuck...
                z_indices_to_plot = 1:length(z_indices);
                if length(z_indices) > max_figures_to_plot
                    plot_chunk = 1;
                    z_indices_to_plot = [(plot_chunk-1)*max_figures_to_plot+1 : min([plot_chunk*max_figures_to_plot, length(z_indices)])];
                end
                for ind = z_indices_to_plot
                    z_index = z_indices(ind);
                    plot_flattening_results(BatchOfData_image_num, z_index, boundary_cell_arr{ind}, boundary_cell_arr_in{ind}, boundary_cell_arr_out{ind}, ...
                        R_disk_vec(ind), seg(:,:,z_index), seg_before_dilate(:,:,z_index), seg_final(:,:,z_index), seg_flattened(:,:,ind), seg_boundary_flattened{ind}, ...
                        im_store(:,:,z_index), im_flattened(:,:,ind), im_boundary_flattened{ind}, show_type);
                end
%                 z_index = 85
%                 load([image_file_path, '\flatten_old\AFACL', num2str(BatchOfData_image_num), '_01_slice_', num2str(z_index), '.mat']);
%                 if flag_good == 1
%                     plot_flattening_results(BatchOfData_image_num, z_index, boundary, boundary_in, boundary_out, R_disk, ...
%                         seg2d, seg2d_before_dilate, seg2d_final, seg_flattened, seg_boundary_flattened, ...
%                         im_2D, im_flattened_clipped, im_boundary_flattened, show_type);
%                 else % flag_good = 0
%                     plot_non_flattened_results(BatchOfData_image_num, z_index, boundary, R_disk, seg2d, seg2d_before_dilate, seg2d_final, im_2D, show_type);
%                 end
    end
end
end_time = datetime('now')
elapsed_time = etime(datevec(end_time), datevec(start_time));
disp(['Elapsed time = ', num2str(floor(elapsed_time/60)),' min, ', num2str(round(mod(elapsed_time,60))), ' sec']);
