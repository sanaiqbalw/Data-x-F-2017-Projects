function curvature = get_curvature(x, y, method)
    if ~isequal(size(x), size(y)) || (sum(size(x)==1) == 0)
        error('x and y should be vectors of the same size!');
    end
    if isequal([x(1) y(1)], [x(end) y(end)]) % the first and the last points are the same
        x = x(1:end-1);
        y = y(1:end-1);
    end
    
    numberOfPoints = length(x);
    curvature = zeros(size(x));
    for t = 1 : numberOfPoints
        if t == 1
            index1 = numberOfPoints;
            index2 = 1;
            index3 = 2;
        elseif t == numberOfPoints
            index1 = numberOfPoints-1;
            index2 = numberOfPoints;
            index3 = 1;
        else
            index1 = t - 1;
            index2 = t;
            index3 = t + 1;
        end
        % Get the 3 points.
        x1 = x(index1);
        y1 = y(index1);
        x2 = x(index2);
        y2 = y(index2);
        x3 = x(index3);
        y3 = y(index3);
        if method == 'Roger'
            % Now call Roger's formula:
            % http://www.mathworks.com/matlabcentral/answers/57194#answer_69185
            curvature(t) = 2*abs((x2-x1).*(y3-y1)-(x3-x1).*(y2-y1)) ./ sqrt(((x2-x1).^2+(y2-y1).^2)*((x3-x1).^2+(y3-y1).^2)*((x3-x2).^2+(y3-y2).^2));
        elseif method == 'angle'
            a = [x1-x2 y1-y2];
            b = [x3-x2 y3-y2];
            a_cross_b = cross([a 0],[b 0]);
            angle_deg = (180/pi)*atan2(a_cross_b(3), dot(a,b));
            if angle_deg > 0
                angle_deg = 180 - angle_deg;
            else
                angle_deg = -180 - angle_deg;
            end
            curvature(t) = angle_deg;
        end
    end
end
