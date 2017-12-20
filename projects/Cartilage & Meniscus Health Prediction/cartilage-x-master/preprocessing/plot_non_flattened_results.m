function plot_non_flattened_results(BatchOfData_image_num, z_index, boundary, R_disk, seg2d, seg2d_before_dilate, seg2d_final, im_2D, show_type)
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
plot(boundary(:, 2), boundary(:, 1), 'g-o', 'LineWidth', LineWidth);
title(['Final segnmentation, R_{disk}=', num2str(R_disk)]);

subplot(plot_rows, plot_cols, 3);
show_image(im_2D, show_type); %axis(axis_vec);
hold on;
plot(boundary(:, 2), boundary(:, 1), 'g-o', 'LineWidth', LineWidth);
title('Original image');
end
