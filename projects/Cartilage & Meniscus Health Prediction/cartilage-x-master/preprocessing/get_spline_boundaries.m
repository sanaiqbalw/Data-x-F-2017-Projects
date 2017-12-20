function [boundary, boundary_in, boundary_out, end_x, end_y, ind_sharp_thin] = get_spline_boundaries(seg2d)
    [B,L] = bwboundaries(seg2d, 'noholes');
    boundary = B{1};
    
    %% Thinning the pieces and finding the endpoints
    seg2d_final_thin = bwmorph(seg2d, 'thin', Inf);
    [end_x, end_y] = find(bwmorph(seg2d_final_thin, 'endpoints'));
    end_x = end_x(1:2); end_y = end_y(1:2); % just for now

    %% Finding the closest neighbor to the endpoints
    ind_sharp_thin = zeros(1,2);
    for ind=1:2 % finding the nearest neighbor on the spline
        [min_val, min_ind] = min((end_x(ind)-boundary(:,1)).^2 +(end_y(ind)-boundary(:,2)).^2);
        ind_sharp_thin(ind) = min_ind;
    end

    %% Splitting the boundary
    % bwboundaries order is always clockwise
    if end_y(2) > end_y(1) % the first edge is on the left
        if ind_sharp_thin(2) > ind_sharp_thin(1) % the boundary starts at the bottom
%             disp('first edge is on the left, boundary starts at the bottom');
            shift = -ind_sharp_thin(1);
            ind_sharp_thin_shifted = ind_sharp_thin(2) - ind_sharp_thin(1);
        else % the boundary starts at the top
%             disp('first edge is on the left, boundary starts at the top');
            shift = size(boundary,1) - ind_sharp_thin(1);
            ind_sharp_thin_shifted = ind_sharp_thin(2) + shift;
        end
    else % the first edge is on the right
        if ind_sharp_thin(2) > ind_sharp_thin(1) % the boundary starts at the bottom
%             disp('first edge is on the right, boundary starts at the bottom');
        else % the boundary starts at the top
%             disp('first edge is on the right, boundary starts at the top');
        end
    end
    boundary_shifted = circshift(boundary, shift, 1);
    % the first index is now at one edge of the boundary
    boundary_in = boundary_shifted(1:ind_sharp_thin_shifted+1, :);
    boundary_in = [boundary_shifted(end, :); boundary_in];
    boundary_out = boundary_shifted(ind_sharp_thin_shifted+1:end, :);
%     boundary_in = boundary_shifted(1:ind_sharp_thin_shifted, :);
%     boundary_out = boundary_shifted(ind_sharp_thin_shifted+1:end, :);
end
