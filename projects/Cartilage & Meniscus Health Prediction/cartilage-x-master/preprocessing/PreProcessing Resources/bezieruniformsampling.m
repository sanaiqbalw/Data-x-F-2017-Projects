function [mOut] = bezieruniformsampling(p,resxy,increment)
% [mOut] = bezieruniformsampling(p,resxy,increment)
% 
% Function to create a Bezier spline with a number of points between control poinst that is proportional to the distance between the control points in question. 
% Based on bezier.m by Kehyang Lee
%
% 
% Inputs:          
% p              - Nx3 matrix with the coordinates of the control points.
% resxy          - Spatial resolution in the row or column direction.
% increment      - A double less than 1 with the resolution of the spline.
% 
% Outputs:
% mOut           - A 3-column array with the coordinates of the points along the spline. 
% 
% Endpoints:      x0, x3, y0, y3
% Control points: x1, x2, y1, y2
%
% Bezier function:
%    x(t) = ax * t^3 + bx * t^2 + cx * t + x0
%    y(t) = ay * t^3 + by * t^2 + cy * t + y0
%    as t = 0~1.
%
%
%
% by 
% Julio Carballido-Gamio
% 2005
% Julio.Carballido@gmail.com
%
% VersionString = '1.01.01'  ;                 % major, minor and source version
% VersionTrack(mfilename,VersionString);       % This call stores the version string for inclusion in outputs  
  
mOut = p(1,:);
for nTemp1 = 1 : size(p,1)-1

    cdist = sqrt(sum((p(nTemp1,:)-p(nTemp1+1,:)).^2))*resxy;
    if cdist==0;
        cdist=0.1;
    end
    if nargin<3
        increment = 0.1/(cdist);%0.1,0.3  0.25
    end
    % Define endpoints
    x0 = p(nTemp1,1);    x3 = p(nTemp1+1,1);
    y0 = p(nTemp1,2);    y3 = p(nTemp1+1,2);

    % Define control points
    if nTemp1 ~= 1
        v13 = p(nTemp1+1,:) - p(nTemp1-1,:);
        v13 = v13 / sqrt(sum(v13.^2));
        d13 = sqrt(sum((p(nTemp1-1,:)-p(nTemp1+1,:)).^2));
        d23 = sqrt(sum((p(nTemp1,:)-p(nTemp1+1,:)).^2));
        d12 = sqrt(sum((p(nTemp1-1,:)-p(nTemp1,:)).^2));
        Del_d = (d23^2 + d13^2 - d12^2) / 2 / d13;

        x1 = p(nTemp1,1) + Del_d * v13(1) / 2;
        y1 = p(nTemp1,2) + Del_d * v13(2) / 2;
    else
        x1 = x0;    y1 = y0;
    end
    if nTemp1 ~= size(p,1)-1
        v31 = p(nTemp1,:) - p(nTemp1+2,:);
        v31 = v31 / sqrt(sum(v31.^2));
        d13 = sqrt(sum((p(nTemp1,:)-p(nTemp1+2,:)).^2));
        d23 = sqrt(sum((p(nTemp1+1,:)-p(nTemp1+2,:)).^2));
        d12 = sqrt(sum((p(nTemp1,:)-p(nTemp1+1,:)).^2));
        Del_d = (d12^2 + d13^2 - d23^2) / 2 / d13;

        x2 = p(nTemp1+1,1) + Del_d * v31(1) / 2;
        y2 = p(nTemp1+1,2) + Del_d * v31(2) / 2;
    else
        x2 = x3;    y2 = y3;
    end

%        plot(x1,y1,'ro',x2,y2,'rx')  % plot control points

    cx = 3*(x1-x0);
    bx = 3*(x2-x1)-cx;
    ax = x3-x0-cx-bx;
    cy = 3*(y1-y0);
    by = 3*(y2-y1)-cy;
    ay = y3-y0-cy-by;

    t=0:increment:1;
    x = polyval([ax,bx,cx,x0],t);
    y = polyval([ay,by,cy,y0],t);
    mOut = [mOut;x',y',p(1,3)*ones(size(x'))];
end

mOut(1,:) = [];
mOut(end+1,:) = p(1,3);
mOut(end,1) = p(end,1);
mOut(end,2) = p(end,2);