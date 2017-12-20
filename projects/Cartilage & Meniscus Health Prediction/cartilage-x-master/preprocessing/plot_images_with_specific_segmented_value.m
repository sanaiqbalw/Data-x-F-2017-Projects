function indices = plot_images_with_specific_segmented_value(im_store, Mask, value_plot, number_of_plots, plot_flag, show_type)

fig_size = [0.66 0.5];
% number_of_plots = 0; % 0 for plotting all
plot_counter = 0;
% value_plot = 1
% value_plot = 2
indices = [];
for image_number = [1:size(Mask,3)]
    seg2d = Mask(:, :, image_number);
    seg2d(seg2d ~= value_plot) = 0;
    if max(max(seg2d)) > 0
        if (number_of_plots  == 0) | (plot_counter < number_of_plots)
            plot_counter = plot_counter + 1;
            indices(plot_counter) = image_number;
            if plot_flag == 1
                figure
                set(gcf,'units','normalized','outerposition',[0.5-fig_size/2 fig_size]);
                subplot(1,2,1);
                show_image(im_store(:, :, image_number), show_type);
                title(['Image number ', num2str(image_number)]);
                subplot(1,2,2);
                show_image(seg2d, show_type);
                title(['Segmented number ', num2str(image_number), ', value = ', num2str(value_plot)]);
            end
        end
    end
end

end