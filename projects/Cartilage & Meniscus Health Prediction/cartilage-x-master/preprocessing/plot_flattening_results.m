function plot_flattening_results(BatchOfData_image_num, z_index, boundary, boundary_in, boundary_out, R_disk, ...
    seg2d, seg2d_before_dilate, seg2d_final, seg_flattened, seg_boundary_flattened, ...
    im_2D, im_flattened_clipped, im_boundary_flattened, show_type)
LineWidth = 1.5;
fig_size = [1 1];
plot_rows = 2; plot_cols = 3;
d = 15;
axis_vec = [min(boundary(:,2))-d max(boundary(:,2))+d min(boundary(:,1))-d max(boundary(:,1))+d];
%         axis_vec = [0 511 0 511];
figure
set(gcf,'units','normalized','outerposition',[0.5-fig_size/2 fig_size]);

subplot(plot_rows, plot_cols, 1);
show_image(seg2d, show_type);
title(['BatchOfData seg. image ', num2str(BatchOfData_image_num), ', z index ', num2str(z_index)]); axis(axis_vec);

subplot(plot_rows, plot_cols, 4);
show_image(seg2d_before_dilate, show_type);
title('Combining with neighbors');

subplot(plot_rows, plot_cols, 2);
show_image(seg2d_final, show_type); axis(axis_vec);
hold on;
plot(boundary_in(:, 2), boundary_in(:, 1), 'r-o',...
        boundary_out(:, 2), boundary_out(:, 1), 'b-o','LineWidth', LineWidth);
title(['Boundary splines after splitting, R_{disk}=', num2str(R_disk)]);

subplot(plot_rows, plot_cols, 5);
d_flattened = 30;
axis_flattened_vec = [min(seg_boundary_flattened(:,1))-d_flattened 
                    max(seg_boundary_flattened(:,1))+d_flattened 
                    min(seg_boundary_flattened(:,2))-d_flattened 
                    max(seg_boundary_flattened(:,2))+d_flattened];
show_image(seg_flattened, show_type); %axis(axis_flattened_vec);
hold on;
plot(seg_boundary_flattened(:, 1), seg_boundary_flattened(:, 2), 'g-o', 'LineWidth', LineWidth);
title('Flattened segmentation');

subplot(plot_rows, plot_cols, 3);
show_image(im_2D, show_type); %axis(axis_vec);
hold on;
plot(boundary_in(:, 2), boundary_in(:, 1), 'r-o',...
        boundary_out(:, 2), boundary_out(:, 1), 'b-o','LineWidth', LineWidth);
title('Original image');

subplot(plot_rows, plot_cols, 6);
show_image(im_flattened_clipped, show_type); %axis(axis_flattened_vec);
hold on;
plot(im_boundary_flattened(:, 1), im_boundary_flattened(:, 2), 'g-o', 'LineWidth', LineWidth);
title('Flattened image');
end
