function [boundary, boundary_in, boundary_out, num_sharp_edges] = get_spline_boundaries(seg2d, method)
    [B,L] = bwboundaries(seg2d, 'noholes');
    boundary = B{1}; % only one object in this simple case
    % flipping the columns to be consistent with the plots order
    boundary = [boundary(:,2) boundary(:,1)];
    
    if ~strcmp(method, 'kosta')
        curvature = get_curvature(boundary(:,1), boundary(:,2), method);
        ind_sharp = find(curvature > 0.7);
    %     figure
    %     plot(curvature, 'b-o', 'LineWidth', 1.5);
    %     set(gca, 'YTick', [-180 : 45 : 180]);
    %     ylim([-90 90]);
    %     grid on;
    %     boundary_in = [];
    %     boundary_out = [];
    %     num_sharp_edges = 0;
    else % method='kosta'
        % calculating derivatives to get the sharp edges
        boundary_diff = [diff(boundary(:,1)) diff(boundary(:,2))]; % each value is -1, 0 or 1
        boundary_diff2 = [diff(boundary_diff(:,1)) diff(boundary_diff(:,2))]; % each value is -1, 0 or 1
        ind_sharp = find(sum(abs(boundary_diff2),2) > 1) + 1; % looking for sharp changes where both 2nd derivatives are non-zero
    end
    num_sharp_edges = length(ind_sharp);
    if num_sharp_edges <= 2
        if num_sharp_edges == 2 % two boundary ends
            boundary = circshift(boundary, -ind_sharp(1), 1);
            ind_sharp = ind_sharp(2) - ind_sharp(1);
        end
        % the first index is at one end of the boundary
        boundary_in = boundary(1:ind_sharp, :);
        boundary_out = boundary(ind_sharp+1:end, :);
    else
        disp(['There are ', num2str(num_sharp_edges), ' sharp changes!! Cannot find in and out boundaries!!']);
        boundary_in = [];
        boundary_out = [];
    end
end
