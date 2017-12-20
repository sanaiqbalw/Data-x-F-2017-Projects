function [flag_good, boundary, boundary_in, boundary_out, end_x, end_y, ind_sharp_thin] = get_spline_boundaries2(seg2d, method)
    
    num_pixels = length(seg2d(seg2d == 1));
    if num_pixels < 100 % the cartilage is too small, the algorithm breaks
        flag_good = 0;
        boundary_in = []; boundary_out = []; end_x = []; end_y = []; ind_sharp_thin = [];
        if num_pixels < 10 % single pixels that were removed by bwmorph(seg2d, 'clean')
            boundary = [0 0];
        else
            [B,L] = bwboundaries(seg2d, 'noholes');
            boundary = B{1};
        end
        return;
    else
        flag_good = 1;
    end
    [B,L] = bwboundaries(seg2d, 'noholes');
    boundary = B{1};
    
    %% Thinning the pieces and finding the endpoints
    seg2d_final_thin = bwmorph(seg2d, 'thin', Inf);
    switch method
        case 'endpoints'
            % pick the two endpoints with the max distance between each other.
            % This method is effective if the cartilage is thin, so the nearest neigbor to the endpoints is actually near the endpoints.
            [end_x, end_y] = find(bwmorph(seg2d_final_thin, 'endpoints'));
            [end_x, end_y] = find_max_distance(end_x, end_y);
        case 'polyfit'
            % fitting a 2nd order polynomial to the thinned cartilage, and picking the points on the boundary that intersect it. 
            % This method is effective if the cartilage is thick enough, so it intersects the polynomial only at the edges. 
            ind2d = find(seg2d_final_thin == 1);
            [thin_x, thin_y] = ind2sub(size(seg2d_final_thin), ind2d);

            p = polyfit(thin_y, thin_x, 2);
%             y1 = [0 : 511];
%             thin_x_polyfit = polyval(p, y1);

            boundary_poly = polyval(p, boundary(:, 2));
            boundary_distance = abs(boundary_poly - boundary(:, 1));
            [val_sorted, ind_sorted] = sort(boundary_distance);
            max_ind_sorted = 4;
            [end_x, end_y] = find_max_distance(boundary(ind_sorted(1:max_ind_sorted), 1), boundary(ind_sorted(1:max_ind_sorted), 2));
            if end_y(1) > end_y(2)
                end_x = flipud(end_x);
                end_y = flipud(end_y);
            end
    end
                    
    %% Finding the closest neighbor to the endpoints
    % For method='polyfit' it does nothing since the "endpoint" is already on the boundary
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
