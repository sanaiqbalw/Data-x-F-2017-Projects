function [ind_x_out, ind_y_out] = find_max_distance(ind_x_in, ind_y_in)
% returns the two points that have the maximum distance between each other
points = [ind_y_in, ind_x_in];
points_distances = dist2inC(points, points);
[max_val_col, ind_row] = max(points_distances);
[max_val_total, ind_col] = max(max_val_col);
max_distance_indices = sort([ind_row(ind_col), ind_col]);
ind_x_out = ind_x_in(max_distance_indices);
ind_y_out = ind_y_in(max_distance_indices);
end
