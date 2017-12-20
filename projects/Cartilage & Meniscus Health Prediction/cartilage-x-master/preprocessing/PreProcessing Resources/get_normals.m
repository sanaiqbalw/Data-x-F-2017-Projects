function [angles,allangles,allpts] = get_normals(sl,donotcreatespline)
% [angles,allangles,allpts] = get_normals(sl,donotcreatespline)
% 
% Function to computes normals from a spline.
% 
% Inputs:
% sl                  - Array of nx2 with (x,y) coordinates of points.
% donotcreatespline   - Flag to tell the program that we are providing the actual spline and not only the control points in sl.
%                       The value is irrelevant as long as is a number.
%
% Outputs:
% angles              - A column vector with the normal angles (radians). 
%                       If sl contained control points only, then it has the normal angles at the control points. Otherwise it is redundant; the same as allangles.
% allangles           - A column vector with the normal angles (radians) at each single point of the spline.
% allpts              - An array with the (x,y) coordnates of all the points in the spline.
% 
% Note: The normal vector of the last point is not provided.
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
displayflag = 0;
%-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
% Get the number of points
npts = size(sl,1);
if nargin==1 % Then create a new spline
   [pts] = bezier([sl(:,1) sl(:,2) ones(npts,1)]);
   pts(:,3) = [];
elseif nargin==2 % Do not create a spline
   [pts] = sl;
end
%-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
% Get the differences between consecutive points
tangentvectorx = diff(pts(:,1));
tangentvectory = diff(pts(:,2));
% Get the length of the tangent vector
normtangent = sqrt(tangentvectorx.^2+tangentvectory.^2);
% Be careful with possible overlapping points
pos0 = find(normtangent==0);
normtangent(pos0) = 1;
% Normalize the x and y components by the length of the tangent vector
tangentvectorx = tangentvectorx./normtangent;
tangentvectory = tangentvectory./normtangent;
%-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
% Get the tangent angle, then sum pi/2 to get the normal one
angles = atan2(tangentvectory,tangentvectorx)+pi/2;
% Change the range from [-pi pi] to [0 2*pi)
angles = angles+(angles<0)*(2*pi);
angles(angles==2*pi) = 0;
%-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
% Deal with possible overlapping points (see above)
angles(pos0) = 0;
%-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
% Keep all the angles
allangles = angles;
%-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
pts(end,:) = [];
newpos = [];
sl(end,:) = [];
npts = npts-1;
if nargin==1 % Match the evaluated spline points and the control points
   newpos = matchingpointsinC(sl(:,1:2),pts);
elseif nargin==2
   newpos = 1:npts; % No need to match
end
% Keep all the points
allpts = pts;
angles = angles(newpos);
%-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
if displayflag
    pts = pts(newpos,:);
    figure;
    hold on 
    plot(pts(:,1),pts(:,2),'r-');
    hold on; 
    quiver(pts(:,1),pts(:,2),cos(angles),sin(angles),1);
    set(gca,'YDir','Reverse');
    figure;
    mydelta = 25;
    hold on;
    plot(allpts(1:mydelta:end,1),allpts(1:mydelta:end,2),'r-');
    quiver(allpts(1:mydelta:end,1),allpts(1:mydelta:end,2),cos(allangles(1:mydelta:end)),sin(allangles(1:mydelta:end)),1);
    set(gca,'YDir','Reverse');
    hold off;
    size(allpts)
    size(allangles)
    axis equal;
end
%-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
