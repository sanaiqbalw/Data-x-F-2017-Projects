function Streamlines = getstreamlines2D(PotentialMap,bspline,aval,mask,displayflag)
% Streamlines = getstreamlines2D(PotentialMap,bspline,aval,mask,displayflag)
%
% 2D function to match bone-cartilage interface points to articular points based on a potential map which results from numerically solving the Laplace's equation.
% 
% Inputs:
% PotentialMap    - 2D array of doubles with the potential map.
% bspline         - 3-column array with the voxel coordinates of the reference spline.
% aval            - Scalar indicating the potential value where the streamlines will end.
% mask            - 2D array of doubles with a binary mask of the ROI.
% displayflag     - displayflag=1 displays a sampled version of the streamlines overlaid on the potential map.
%                   displayflag=0 displays nothing (default value).
%
% Outputs:
% Streamlines     - Cell array with the coordinates of the streamlines for each point in bspline [row column].
%
%
%
% by
% Julio Carballido-Gamio
% 2010
% Julio.Carballido@gmail.com
%

% Check inputs
if ~exist('PotentialMap','var') || ~exist('bspline','var') || ~exist('aval','var') || ~exist('mask','var')
    Disp('At least 4 inputs are required. Streamlines were not computed.');
    Streamlines = [];   
    return;         
end
if ~exist('displayflag','var'),      displayflag = 0;     end

%-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
% Get the image size
[w1p w2p] = size(PotentialMap);
% Get two-point difference gradients
% Gx = (circshift(A,[0 1])-circshift(A,[0 -1]))/2;
% Gy = (circshift(A,[1 0])-circshift(A,[-1 0]))/2;
% Or use MATLAB function
[Gx Gy] = gradient(PotentialMap);
Gx = Gx.*mask;
Gy = Gy.*mask;
% Normalize the gradients
Gm = sqrt(Gx.^2+Gy.^2);
Gmno0 = Gm>0;
Gx(Gmno0) = Gx(Gmno0)./Gm(Gmno0);
Gy(Gmno0) = Gy(Gmno0)./Gm(Gmno0);
% Get tangents for the borders which do not always look ok
cartROI = bwboundaries(mask);
cartROI = cartROI{1};
pos = sub2ind([w1p w2p],cartROI(:,1),cartROI(:,2));
Gx(pos) = 0;
Gy(pos) = 0;
Gxm = imfilter(Gx,fspecial('average',[3 3]));
Gym = imfilter(Gy,fspecial('average',[3 3]));
Gx(pos) = Gxm(pos);
Gy(pos) = Gym(pos);
% Get number of points in the bone spline
nptsb = size(bspline,1);   

%-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
% Use Eurler's method to get the streamlines
% Prepare a cell to store the streamlines
Streamlines = cell(nptsb,1);
% Store the first and last points
Streamlines{1} = bspline(1,:);
Streamlines{nptsb} = bspline(end,:);
% Establish a small step size
mydelta = 0.1;
for npt=2:nptsb-1 % Because we are assuming that the end-points are the same in the bspline and aspline
    Streamlines{npt} = bspline(npt,:);
    isarticular = 0;
    while isarticular==0
        % Get current row and column
        crow = Streamlines{npt}(end,1);
        ccol = Streamlines{npt}(end,2);
        % Round the current row and column to find the tangents
        rcrow = round(crow);
        rccol = round(ccol);
        % Get the angle of the vector
        theta = atan2(Gy(rcrow,rccol),Gx(rcrow,rccol));
        % Get the new row and new column
        newr = crow+mydelta*sin(theta);
        newc = ccol+mydelta*cos(theta);
        % Check if the streamline has reached the background
        rnewr = round(newr);
        rnewc = round(newc);
        if (mask(rnewr,rnewc)==0)
            isarticular = 1;
            % Check if it actually reached the articular surface
            lastval = PotentialMap(round(crow),round(ccol));
            if lastval~=aval % Discard the streamline
                Streamlines{npt} = [];
            end
        else
            % Stack the current coordinate to the current streamline
            Streamlines{npt} = [Streamlines{npt}; [newr newc]];
        end
    end
end

%-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
% Display
if displayflag
    if 0
        figure;
        PotMap = PotentialMap;
        PotMap = PotMap.*mask;
        PotMap = PotMap-0.999999999;
        PotMap(PotMap<0) = 0;
        imagesc(PotMap); axis equal; caxis([0 1]); impixelinfo; colormap(hot);
        hold on;  
        for npt=1:5:nptsb
            if isempty(Streamlines{npt})
                continue;
            end
            plot(Streamlines{npt}(:,2),Streamlines{npt}(:,1),'k');
        end
        drawnow;
        hold off;
    end
    %----------------------------------------------------------------------------------------%
    % Display the image as RGB
    if 1
       figure('NumberTitle','off', ...
                 'Name','Streamlines on Potential Map');
        npixels = w1p*w2p;
        PotMap = PotentialMap;
        PotMap = PotMap-0.999999;
        PotMap(PotMap<0) = 0;
        cmap = hot(256);
        [indmap] = gray2ind(PotMap,256); 
        ima3 = zeros(w1p,w2p,3);
        for n=1:npixels
            ima3(n)=cmap(double(indmap(n)+1),1); 
            ima3(n+npixels)=cmap(double(indmap(n)+1),2); 
            ima3(n+2*npixels)=cmap(double(indmap(n)+1),3); 
        end    
        pos0 = find(mask==0);
        ima3(pos0) = 0;
        ima3(npixels+pos0) = 0.25;
        ima3(2*npixels+pos0) = 1;
        imagesc(ima3);  axis equal; 
        hold on;  
        for npt=1:5:nptsb
            if isempty(Streamlines{npt})
                continue;
            end
            plot(Streamlines{npt}(:,2),Streamlines{npt}(:,1),'k');
        end
        drawnow;
    end
end