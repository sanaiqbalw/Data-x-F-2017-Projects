clear all
close all
clc
LineWidth = 1.5;
addpath('PreProcessing Resources');

splines_type = 'get_spline_boundaries';
% splines_type = 'load from file'; % a good spline, not sure which image does it match to

% show_type = 'imshow';
% show_type = 'image';
show_type = 'imagesc';

bad_guys = [2006, 60];
% bad_guys = [2008, 57];
% bad_guys = [2009, 138];.

% bad_guys = [2008, 90];
% bad_guys = [2013, 42];
% bad_guys = [2014, 189];

% for BatchOfData_image_num = [[2005:2009] [2011:2022] [3001:3003] [3005:3008]]
% for BatchOfData_image_num = [2005]
for BatchOfData_image_num = [bad_guys(1)]
    disp(['Processing image ', num2str(BatchOfData_image_num),' ...']);
    load(['D:\Berkeley Google Drive\Cartilage-X\BatchOfData\seg\AFACL', num2str(BatchOfData_image_num), '_01.mat']); % pred_con_vol
    Mask = pred_con_vol;
%     man_image_num = 44;
%     load(['D:\Google Drive\Berkeley\Courses\Data X IEOR 290\Project\Cartilage-X\PreProcessing Resources\manualExamplesforKosta\seg_', num2str(man_image_num), '.mat']); % Mask
    
    % getting the z indices of the masks with the right value
    value_plot = 1; % MFC, cartilage, medial femoral condyle
    plot_flag = 0;
    disp(['Image ', num2str(BatchOfData_image_num),' has ', num2str(size(Mask,3)), ' slices']);
    disp(['Indices of slices that have segmented values of ', num2str(value_plot), ':']);
%     z_indices = plot_images_with_specific_segmented_value(Mask, Mask, value_plot, 0, plot_flag, show_type)
%     z_indices = [85];
    z_indices = [bad_guys(2)];

    R_disk_vec = zeros(size(z_indices));
    for non_zero_ind = 1:length(z_indices)
        z_index = z_indices(non_zero_ind);
        disp(['Processing z index ', num2str(z_index),' ...']);
        seg2d = Mask(:, :, z_index);
        seg2d(seg2d ~= value_plot) = 0;
        seg2d = bwmorph(seg2d, 'clean'); % remove single pixels
        seg2d_fill = imfill(seg2d, 'holes'); % fill the holes
        seg2d_final = seg2d_fill;
        [B,L] = bwboundaries(seg2d_final, 'noholes'); % find the spline boundaries
        R_disk = 0;
        flag_imclose = 0;
        if length(B) > 1
            flag_imclose = 1;
            close_seg2d_fill = imclose(seg2d_fill, strel('disk', 50, 6));  % combine the neighbors
            seg2d_final = close_seg2d_fill;
            [B,L] = bwboundaries(seg2d_final, 'noholes'); % find the spline boundaries          
            while length(B) > 1 % dilate if there is more than one piece
                R_disk = R_disk + 1;
                seg2d_final = imdilate(seg2d_final, strel('disk', R_disk)); 
                [B,L] = bwboundaries(seg2d_final, 'noholes');
            end
        end
        R_disk_vec(non_zero_ind) = R_disk;
%         if length(B) == 1 % only one piece
%             
%         else

%         end
%         seg2d_final = seg2d_fill;
%         R_disk = 0;

%         if R_disk == 0
        if R_disk == R_disk
            image_to_flatten = double(seg2d_final);

            switch splines_type
                case 'get_spline_boundaries'
                    if R_disk < 6
                        method = 'endpoints';
                    else
                        method = 'polyfit';
                    end
%                     [boundary, boundary_in, boundary_out, end_x, end_y, ind_sharp_thin] = get_spline_boundaries_debug(seg2d_final); % in _debug() the columns of the boundaries are flipped
%                     [boundary, boundary_in, boundary_out, end_x, end_y, ind_sharp_thin] = get_spline_boundaries(seg2d_final);
%                     [boundary, boundary_in, boundary_out, end_x, end_y, ind_sharp_thin] = get_spline_boundaries2(seg2d_final, method);
                    [flag_good, boundary, boundary_in, boundary_out, end_x, end_y, ind_sharp_thin] = get_spline_boundaries2(seg2d_final, method);
                    
%                     [B,L] = bwboundaries(seg2d_final, 'noholes');
%                     boundary = B{1};
%                     figure
%                     show_image(seg2d_final, show_type);
%                     hold on;
%                     plot(boundary(:, 2), boundary(:, 1), 'g-o', 'LineWidth', LineWidth);
                    
%                     [B,L] = bwboundaries(seg2d_final, 'noholes');
%                     boundary = B{1};
%                     seg2d_final_thin = bwmorph(seg2d_final, 'thin', Inf);
%                     switch method
%                         case 'endpoints'
%                             [end_x, end_y] = find(bwmorph(seg2d_final_thin, 'endpoints'));
%                             [end_x, end_y] = find_max_distance(end_x, end_y);
% %                             endpoints = [end_y, end_x];
% %                             % finding the two endpoints with the max distance between each other
% %                             endpoints_distances = dist2inC(endpoints, endpoints);
% %                             [max_val_col, ind_row] = max(endpoints_distances);
% %                             [max_val_total, ind_col] = max(max_val_col);
% %                             max_distance_indices = sort([ind_row(ind_col), ind_col]);
% %                             end_x = end_x(max_distance_indices);
% %                             end_y = end_y(max_distance_indices);
%                             
%                             figure
%                             show_image(seg2d_final_thin, show_type);
%                             hold on;
%                             plot(boundary(:, 2), boundary(:, 1), 'g-o', 'LineWidth', LineWidth);
%                             hold on;
%                             LineWidth_ends = 4;
%                             plot(end_y, end_x, 'mo', 'LineWidth', LineWidth_ends);
%                             
%                         case 'polyfit'
%                             ind2d = find(seg2d_final_thin == 1);
%                             [thin_x, thin_y] = ind2sub(size(seg2d_final_thin), ind2d);
% 
%                             p = polyfit(thin_y, thin_x, 2);
%                             y1 = [0 : 511];
%                             thin_x_polyfit = polyval(p, y1);
%                             
%                             boundary_poly = polyval(p, boundary(:, 2));
%                             boundary_distance = abs(boundary_poly - boundary(:, 1));
%                             [val_sorted, ind_sorted] = sort(boundary_distance);
%                             max_ind_sorted = 4;
%                             [end_x, end_y] = find_max_distance(boundary(ind_sorted(1:max_ind_sorted), 1), boundary(ind_sorted(1:max_ind_sorted), 2));
%                             
%                             figure
%                             show_image(seg2d_final_thin, show_type);
%                             hold on;
%                             plot(boundary(:, 2), boundary(:, 1), 'g-o', 'LineWidth', LineWidth);
%                             hold on;
%                             plot(thin_y, thin_x, 'r*', 'LineWidth', LineWidth);
%                             hold on;
%                             plot(y1, thin_x_polyfit, 'g-', 'LineWidth', LineWidth);
%                             hold on;
%                             LineWidth_ends = 4;
%                             plot(end_y, end_x, 'mo', 'LineWidth', LineWidth_ends);
%                     end
                    

                    
%                     boundary_in_auto = boundary_in;
%                     boundary_out_auto = boundary_out;
%                     save('auto', 'boundary_in_auto', 'boundary_out_auto');

%                     B = bwboundaries(seg2d_final,'noholes');
%                     Bnocell = B{1,1};
%                     ind1 = 10;
%                     ind2 = 324;
%                     splineas = Bnocell(ind1:ind2, :);
%                     splinebs1 = Bnocell(1:ind1,:);
%                     splinebs2 = Bnocell(ind2:end,:);
%                     splinebs = vertcat(splinebs2,splinebs1);
%                     boundary_in = splineas;
%                     boundary_out = splinebs;
%                     boundary = [boundary_in; boundary_out];
% %                     boundary_in_manual = boundary_in;
% %                     boundary_out_manual = boundary_out;
% %                     save('manual', 'boundary_in_manual', 'boundary_out_manual');

%                     myinfotosave = get_myinfotosave(image_to_flatten, boundary_in, boundary_out);
                case 'load from file'
                    load('D:\Google Drive\Berkeley\Courses\Data X IEOR 290\Project\Cartilage-X\PreProcessing Resources\manualExamplesforKosta\ACL037_130222_E318_CUBE_trans_contra_LFC.mat');
                    slice_num = 21;
                    myinfotosave.w3 = 1;
                    myinfotosave.dataperslice = {myinfotosave.dataperslice{slice_num}};
                    boundary_in = [myinfotosave.dataperslice{1}.xcoordinatesspl{1} myinfotosave.dataperslice{1}.ycoordinatesspl{1}];
                    boundary_out = [myinfotosave.dataperslice{1}.xcoordinatesspl{2} myinfotosave.dataperslice{1}.ycoordinatesspl{2}];
                    boundary = [boundary_in; boundary_out];
            end
            
%             figure;
%             show_image(image_to_flatten, show_type);
%             hold on;
%             plot(boundary_in(:,2), boundary_in(:,1), 'r-o', boundary_out(:,2), boundary_out(:,1), 'b-o', 'LineWidth', LineWidth);
            
            fig_size = [1 1];
            plot_rows = 2; plot_cols = 3;
            d = 15;
            axis_vec = [min(boundary(:,2))-d max(boundary(:,2))+d min(boundary(:,1))-d max(boundary(:,1))+d];
%             axis_vec = [0 511 0 511];
            figure
            set(gcf,'units','normalized','outerposition',[0.5-fig_size/2 fig_size]);
            subplot(plot_rows, plot_cols, 1);
            show_image(seg2d, show_type); title(['BatchOfData image ', num2str(BatchOfData_image_num), ', z index ', num2str(z_index)]); axis(axis_vec);

            subplot(plot_rows, plot_cols, 2);
            if flag_imclose == 0
                show_image(seg2d_fill, show_type); title('Removing single pixels and filling the holes');
            else
                show_image(close_seg2d_fill, show_type); title('Removing single pixels, filling the holes and combining with neighbors');
            end
            axis(axis_vec);

            subplot(plot_rows, plot_cols, 3);
            show_image(seg2d_final, show_type); axis(axis_vec);
            hold on;
            plot(boundary(:, 2), boundary(:, 1), 'g-o', 'LineWidth', LineWidth);
            title(['R_{disk}=', num2str(R_disk), ', finding edges to split the boundary']);
            if flag_good == 1
                hold on;
                LineWidth_ends = 4;
                plot(end_y, end_x, 'mo', 'LineWidth', LineWidth_ends);
            end
%             hold on;
%             plot(boundary(ind_sharp_thin(1),2), boundary(ind_sharp_thin(1),1), 'rs', ...
%                     boundary(ind_sharp_thin(2),2), boundary(ind_sharp_thin(2),1), 'bs', 'LineWidth', LineWidth_ends);
            
            if flag_good == 1
                subplot(plot_rows, plot_cols, 4);
                show_image(seg2d_final, show_type); axis(axis_vec);
                hold on;
                plot(boundary(ind_sharp_thin(1),2), boundary(ind_sharp_thin(1),1), 'rs', ...
                        boundary(ind_sharp_thin(2),2), boundary(ind_sharp_thin(2),1), 'bs', 'LineWidth', LineWidth_ends);
                hold on;
                plot(boundary_in(:, 2), boundary_in(:, 1), 'r-o',...
                        boundary_out(:, 2), boundary_out(:, 1), 'b-o','LineWidth', LineWidth);
                title('Boundary splines after splitting');
            end

            if flag_good == 1
                [image_flattened, boundary_flattened] = flat_2D_cartilage_warping_claudia(image_to_flatten, boundary_in, boundary_out, 0);
                d_flattened = 30;
                axis_flattened_vec = [min(boundary_flattened(:,1))-d_flattened 
                                    max(boundary_flattened(:,1))+d_flattened 
                                    min(boundary_flattened(:,2))-d_flattened 
                                    max(boundary_flattened(:,2))+d_flattened];

                subplot(plot_rows, plot_cols, 5);
                show_image(image_flattened, show_type); axis(axis_flattened_vec);
                hold on;
                plot(boundary_flattened(:, 1), boundary_flattened(:, 2), 'g-o', 'LineWidth', LineWidth);
            end

            
%             matching_method = 3; % 1 - Minimum euclidean distances; 2 - Normal vectors; 3 - Laplacian equation
%             noffsets = 5;
%             isboneref = 1; % the reference for flattening, i.e. what will look like a straight line. isboneref=1: bone-cartilage interface, isboneref=0: articular surface
%             T2Th = []; % a scalar indicating that all Ima values greater than T2Th should be cropped to T2Th. T2Th=[] does not crop Ima.
%             displayflag = 0; % displayflag=1 Displays partial results, displayflag=0 Displays nothing.
%             disp(['noffsets=', num2str(noffsets), ', isboneref=', num2str(isboneref),  ', T2Th=', num2str(T2Th)]);
% %             disp(['Running flat_2D_cartilage_warping() with matching_method=', num2str(matching_method),' ...']);
%             tic_start = tic;
%             [fIma, fmyinfotosave, MeanT2OrigFlat, StdT2OrigFlat, Thicks] = flat_2D_cartilage_warping(image_to_flatten, myinfotosave, matching_method, noffsets, isboneref, T2Th, displayflag);
%             elapsed_time = toc(tic_start); disp(['Elapsed toc time = ', num2str(floor(elapsed_time/60)),' min, ', num2str(round(mod(elapsed_time,60))), ' sec']);
%             disp('Flattened image size:'); disp(size(fIma{1}));
%             disp('Replacing all NaNs with zeros');
%             fIma_index = 1;
%             fIma{fIma_index}(isnan(fIma{fIma_index})) = 0;
% 
%             subplot(plot_rows, plot_cols, 5);
%             show_image(fIma{fIma_index}, show_type);
        end
    end
%     figure
%     stem(z_indices, R_disk_vec, '-o', 'LineWidth', LineWidth);
%     xlabel('z index'); ylabel('R_{disk}');
%     grid on;
end
