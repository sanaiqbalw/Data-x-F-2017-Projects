clear all
close all
clc
LineWidth = 1.5;
addpath('PreProcessing Resources');

%% 
% This is the outline for a (new) version of Laplace transform for cartilage 
% flattening. 
% 
% 1st option is Julio's 2D Laplace transform
% 
%     input:
% 
%         mask: 2D array of doubles, mask has a border section with values 
% equal to n, and another border section with values equal to m where m~=n
% 
%     output:
% 
%         Potential map: 2D array of doubles, same size as mask, with the 
% numerical solution to Laplace's Equation
% 
% 2nd option is MATLAB's regionfill algorithm
% 
%     input: 
% 
%         mask: binary image identifying areas to fill with Laplace solutions
% 
%         image: 2D numeric array, grayscale image, at lease3x3 or larger, 
% same size as mask
% 
%     output:
% 
%         J: filled region in image
% 
% Key differences between the two, Julio's takes in splines (assigned values 
% of 1 and 2) and no other info while MATLAB takes in boundary information of 
% image itself and uses these as boundary conditions
% 
% Create synthetic instance in Julio's format, run through both algorithms

man_image_num = 44;
load(['D:\Google Drive\Berkeley\Courses\Data X IEOR 290\Project\Cartilage-X\PreProcessing Resources\manualExamplesforKosta\seg_', num2str(man_image_num), '.mat']); % Mask
num_start = 20
testmask = Mask(:, :, num_start);
value_plot = 1;
testmask(testmask ~= value_plot) = 0;

testmask(testmask==2) = 0;
imshow(testmask)
B = bwboundaries(testmask,'noholes')
Bnocell = B{1,1};

plot(Bnocell(:,1), Bnocell(:,2))
%% 
% From segmentation to spline, breakdown of upper spline and lower spline 
% performed by finding extreme changes in slope, for now, extremes found manually 
% and broken down into spline bone, spline articular surface.

ind1 = 10;
ind2 = 324;
splineas = Bnocell(ind1:ind2, :);
splinebs1 = Bnocell(1:ind1,:);
splinebs2 = Bnocell(ind2:end,:);
% splineas = Bnocell(7:358, :);
% splinebs1 = Bnocell(1:7,:);
% splinebs2 = Bnocell(358:end,:);
splinebs = vertcat(splinebs2,splinebs1);

figure
plot(splinebs(:,1), splinebs(:,2), 'b', splineas(:,1), splineas(:,2), 'r');
legend('splinebs','splineas');

figure
imagesc(testmask);
hold on;
plot(splinebs(:,2), splinebs(:,1), 'b-o', splineas(:,2), splineas(:,1), 'r-o', 'LineWidth', LineWidth);
legend('splinebs','splineas');

splineas = Bnocell(7:358, :);
splinebs1 = Bnocell(1:7,:);
splinebs2 = Bnocell(358:end,:);
splinebs = vertcat(splinebs2,splinebs1);

plot(splinebs(:,1), splinebs(:,2))
plot(splineas(:,1), splineas(:,2))
%% 
% Up to now: spline articular surface, spline bone surface. These are simply 
% an array of coordinates where the spline lives. In Julio's code, these splines 
% are assinged a value, bone surface assigned a value of 1, articular surface 
% a value of 2.

grid = zeros(size(testmask));

for i = 1:length(splinebs)
    grid(splinebs(i,1),splinebs(i,2)) = 1;
end

for i = 1:length(splineas)
    grid(splineas(i,1), splineas(i,2)) = 2;
end

figure
imshow(grid)
%% 
% Grid is ready to be used as an input into SolveLaplaceEquation2D.m code.

potentialMap = SolveLaplaceEquation2D(grid,0);

imshow(potentialMap)
%% 
% Now only visualize potential map on area of interest (original mask or 
% this new mask created from the splines)

spline_mask = grid;
spline_mask(spline_mask>0) = 1;
spline_mask = imfill(spline_mask,'holes');
imagesc(potentialMap.*spline_mask)
%% 
% Laplacian equation filled in the values between two spline surfaces.
% 
% Next steps: 1. Get cartilage contour 2. Get first endpoint 3. Get second 
% endpoint 4. Get bone border 5. Get articular border 6. Make them share endpoints
% 
% This was done manually when separating out splines?coordinates should still 
% be the same.
% 
% Then, smooth out splines and get matchings between bone and articular surface. 
% Why not smooth before Laplacian 2D?

splinebs = [smooth(splinebs(:,1),7) smooth(splinebs(:,2),7)];
splineas = [smooth(splineas(:,1),7) smooth(splineas(:,2),7)];
figure
plot(splinebs(:,2), splinebs(:,1))
set(gca,'YDir','Reverse');  axis equal; hold on;
plot(splineas(:,2), splineas(:,1))
set(gca,'YDir','Reverse');  axis equal;
%% 
% "Get matchings between the bone and articular surfaces", bone surface 
% chosen as a reference since edema/other findings will be taken from there
% 
% This calls on function getstreamlines2D.m, which matches bone to articular 
% points based on a potential map
% 
%     inputs: 
% 
%         potentialMap: 2D array of doubles with the potentialMap (masked 
% or non masked?)
% 
%         bspline: 2column array with the coordinates of the reference spline 
% 
%         aval: scalar indicating potential value where streamlines will 
% end (2)
% 
%         mask: 2D array of doubles with a binary mask of the ROI
% 
%     outputs:
% 
%         streamlines:  cell array with coordinates of the streamlines for 
% each point in bspline

streamlines = getstreamlines2D(potentialMap,splinebs,2,spline_mask,1);
%% 
% Potential starting pointStreamlines from potential map! Get the number 
% of streamlines, prepare variables for new splines and thickness value.
%%
nstreamlines = length(streamlines)
rstreamlines = 1;

nsplinebs = [];
nsplineas = [];
r = [];

for npt=1:nstreamlines
    if isempty(streamlines{npt})
        continue;
    end
    nsplinebs = [nsplinebs; streamlines{npt}(1,:)];
    nsplineas = [nsplineas; streamlines{npt}(end,:)];
    
    if rstreamlines
        distba = diff(streamlines{npt},1,1);
        distba = sqrt(sum(distba.^2,2));
        cumdistba = cumsum(distba);
        cumdistba = [0; cumdistba];
        r = [r; cumdistba(end)];
    end
end
%% 
% Now we have splines, cumulative thickness from streamlines and the geodesic 
% length of the bone cartilage interface is calculated
%%
distb = diff(nsplinebs,1,1);
distb = sqrt(sum(distb.^2,2));
cumdistb = cumsum(distb);
cumdistb = [0; cumdistb];
lengthb = cumdistb(end);
%% 
%  Get the distance segments, sampling rate 3

cdist = 0:3:lengthb;
diffend = abs(cdist(end)-lengthb);
if diffend<mean(r)/2
    cdist(end) = lengthb;
else
    cdist = [cdist lengthb];
end
cdist = cdist';
%% 
% Get the spline landmarks on the bone-cartilage interface, get the number 
% of landmarks
% 
% Calls on matchingpointsinC, which finds, for each row in Data1, the closest 
% match in Data2 using squared Euclidean distances
% 
%     input: two inputs Data1, Data2
% 
%     output: column vector with the same number of rows as Data1 with row 
% indices indicating best match in data2
% 
% For functions that call on things that require distinC, pdist2(a,b,'euclidean') 
% performs just as well
%%
% Get the spline landmarks on the bone-cartilage interface
% matchcd = matchpoints(cdist,cumdistb,1);
matchcd = matchpointsinC(cdist,cumdistb);
cland = nsplinebs(matchcd,:);
% Get the number of landmarks in the bone cartilage interface
nlands = length(cland);
%% 
% Flatten the cartilage wrt the first spline point in b (fptb), establishes 
% target landmarks for bone interface, on a horizontal line at the same "y" than 
% that of fptb
% 
% If not horizontal, can be vertical such that its on a line at the same 
% "x" (more applicable for other bone compartments)
% 
% Tlands_direction calculated in other situations, will be assinged a 1 for 
% femoral compartments and -1 for tibia compartments
%%
fptb = nsplinebs(1,:);
mainorientation = 'Horizontal';
Tlands_direction = 1;

if strcmp(mainorientation,'Horizontal')
    Tland = zeros(nlands,1);
    Tland(:,1) = fptb(1,1);
    Tland(:,2) = fptb(1,2);
    Tland(:,1) = Tland(:,1)+Tlands_direction*cdist;
else
    Tland = zeros(nlands,1);
    Tland(:,1) = fptb(1,1);
    Tland(:,2) = fptb(1,2);
    Tland(:,2) = Tland(:,2)+Tlands_direction*cdist;
end

Tlandb = Tland;
%% 
% Now the same process is done for the articular surface

cland = [cland; nsplineas(matchcd,:)];
Tland =  [Tland; zeros(nlands,2)];
if strcmp(mainorientation,'Horizontal')
    Tland(nlands+1:end,1) = Tland(1:nlands,1);
    Tland(nlands+1:end,2) = Tland(1:nlands,2)+Tlands_direction*r(matchcd);
    
else
    Tland(nlands+1:end,2) = Tland(1:nlands,2);
    Tland(nlands+1:end,1) = Tland(1:nlands,1)-Tlands_direction*r(matchcd);
end
Tlanda = Tland(nlands+1:end,:);
%% 
% Remove repeated points to avoid problems with warping (is this necessary?)
%%
cland(1,:) = [];
Tland(1,:) = [];
cland(end,:) = [];
Tland(end,:) = [];
cland = round(cland);
%Tlanda = round(Tlanda)
%% 
% Get Fornefettcoefficients and applying them, this is used to do backward 
% mapping, 
% 
% using getFornefettcoefficients.m
% 
%     inputs:
% 
%         Sland: 2D array with set of source landmarks, each row represents 
% coordinates of a landmark
% 
%         Tland: 2D array with set of target landmarks, each row represents 
% coordinates of a landmark
% 
%             a: scalar that controls the influence of the landmarks in their 
% corresponding neighbors
% 
% using applyForneffetcoeffs.m
% 
%     inputs: 
% 
%         Scoords: 2D array with coordinates to be warped
% 
%         Slandmarks: 2D array with set of source landmarks where each row 
% represents the coordinates of a landmark
% 
%         alphas: array of same size as Slandmarks with the corresponding 
% coefficients for warping
% 
%         a: scalar that controls the influence of the landmarks in their 
% corresponding neighbors
% 
%     output: targetcoords, transformed coordinates
% 
% Need to establish a large support "a" for warping for a smooth deformation
% 
% Get the warping parameters and the coordinates to be warped
% 
% THIS IS WHERE THINGS START TO HAVE ISSUES


% cland
% Tland
[alphasv,av] = getFornefettcoefficients(cland,Tland,[]);

[wsplineb] = applyForneffetcoeffs(nsplinebs,cland,alphasv,av);
[wsplinea] = applyForneffetcoeffs(nsplineas,cland,alphasv,av);

[alphas,a] = getFornefettcoefficients(Tland,cland,[]);

maxx = round(max(Tland(:,1)))+10;
maxy = round(max(Tland(:,2)))+10;
[tx, ty] = ndgrid(1:maxx,1:maxy);
tx = tx(:);
ty = ty(:);
%% 
% Warp the coordinates that include the ROI

[trc] = applyForneffetcoeffs([tx ty],Tland,alphas,a);
%% 
% 
% 
% Getting the original coordinates, obtaining the pixel values of the coordinates 
% that include the ROI, backward mapping and interpolation
%%
% img = load('/Users/talairach/Desktop/ML_examples/mri_45.mat');
% nmap = img.im_store(100:400,100:500,22);
load(['D:\Google Drive\Berkeley\Courses\Data X IEOR 290\Project\Cartilage-X\PreProcessing Resources\manualExamplesforKosta\mri_', num2str(man_image_num), '.mat']);
% nmap = img.im_store(100:400, 100:500, num_start);
% nmap = im_store(:, :, num_start);
nmap = testmask;
% img = load('/Users/talairach/Desktop/ML_examples/mri_45.mat');
% nmap = img.im_store(100:400,100:500,22);
imshow(nmap)

[otx,oty] = meshgrid(1:size(testmask,2),1:size(testmask,1));
fmap = interp2(otx,oty,nmap,reshape(trc(:,2),maxx,maxy),reshape(trc(:,1),maxx,maxy),'spline');
fmap = fmap';
imagesc(fmap,[0 1]);
%% 
% Trying to apply coeff to a large portion

[alphas2,a2] = getFornefettcoefficients(Tland,cland,[]);
maxx2 = round(max(Tland(:,1)))+100;
maxy2 = round(max(Tland(:,2)))+100;
[tx2, ty2] = ndgrid(1:maxx2,1:maxy2);

imagesc(fmap,[0 1])
%% 
% Trying to apply coeff to a large portion
%%
[alphas2,a2] = getFornefettcoefficients(Tland,cland,[]);
maxx2 = round(max(Tland(:,1)))+100
maxy2 = round(max(Tland(:,2)))+100
[tx2, ty2] = ndgrid(1:maxx2,1:maxy2)

tx2 = tx2(:);
ty2 = ty2(:);
[trc2] = applyForneffetcoeffs([tx2 ty2],Tland,alphas2,a2);
[otx2,oty2] = meshgrid(1:size(testmask,2),1:size(testmask,1));
fmap2 = interp2(otx2,oty2,nmap,reshape(trc2(:,2),maxx2,maxy2),reshape(trc2(:,1),maxx2,maxy2),'spline');
fmap2 = fmap2';
%%
imagesc(fmap2,[0 1]);
%% 
% Get the flattened cartilage mask and crop, use bezieruniformsampling.m
% 
%     inputs: 
% 
%         p:Nx3 matrix with the coordinates of the control points
% 
%         resxy: spatial resolution in the row or column direction
% 
%     outputs:
% 
%         mOut: a three column array with coordinates of the points along 
% the spline
%%
% The following 2 variables are to compare T2 global means of original and flattened cartilages
oT2means = [];
fT2means = [];
% The following 2 variables are to compare T2 global stds of original and flattened cartilages
oT2stds = [];
fT2stds = [];

cmask = poly2mask([nsplinebs(:,1); flipdim(nsplineas(2:end-1,1),1)],[nsplinebs(:,2); flipdim(nsplineas(2:end-1,2),1)],size(testmask,1),size(testmask,2));
% Mask the map
cmap = cmask.*nmap;
validpts = cmask>0;
oT2means = [oT2means; mean(cmap(validpts))];
oT2stds = [oT2stds; std(cmap(validpts))];

deltax = 1;%0.2734;
Tsplineb = bezieruniformsampling([Tlandb ones(size(Tlandb,1),1)],deltax);
Tsplinea = bezieruniformsampling([Tlanda ones(size(Tlanda,1),1)],deltax);
wmask = poly2mask([Tsplineb(:,1); Tsplinea(:,1)],[Tsplineb(:,2); Tsplinea(:,2)],maxy,maxx);
% Mask the flatten map

wmap = wmask.*fmap;
wmap2=wmap;
% Avoid Nans
wmap(isnan(wmap)) = 0;
% Get the mean of the flattened cartilage
validpts = wmask>0;
fT2means = [fT2means; mean(wmap(validpts))];
fT2stds = [fT2stds; std(wmap(validpts))];

%minr = round(min(Tland(:,2)))-1;   if minr<1,      minr=1;         end
%maxr = round(max(Tland(:,2)))+1;
%minc = round(min(Tland(:,1)))-1;   if minc<1,      minc=1;         end
%maxc = round(max(Tland(:,1)))+1;
%wmap = wmap(minr:maxr,minc:maxc);
%%
figure
imagesc(fmap,[0 1])
figure
imagesc(flipdim(wmap,1),[0 1])
set(gca,'XDir','Reverse');  axis equal;
%%
% Plot the bone spline
figure;
imagesc(nmap,[0 1])

plot(nsplinebs(:,2),nsplinebs(:,1),'g-'); hold on;
set(gca,'YDir','Reverse');  axis equal;
% Plot the articular spline
plot(nsplineas(:,2),nsplineas(:,1),'r-'); hold on;
set(gca,'YDir','Reverse');
% Plot source landmarks
plot(cland(:,1),cland(:,2),'g.'); hold on;
% Plot target landmarks
plot(Tland(:,1),Tland(:,2),'b.'); hold on;
% Plot the warped-bone spline
plot(wsplineb(:,1),wsplineb(:,2),'r-'); hold on; axis equal;
% Plot the warped-articular spline
plot(wsplinea(:,1),wsplinea(:,2),'b-');hold on;

%figure;
%mod_wsplineb = wsplineb(,:);
%mod_wsplinea = wsplinea(,:);

title(['Spline:']);

% Display the masked map
figure; imagesc(nmap,[0 1]); hold on; % %caxis(gca,[0 100]); axis equal;
plot(nsplinebs(:,2),nsplinebs(:,1),'g-', 'LineWidth', LineWidth); hold on;
set(gca,'YDir','Reverse');  axis equal;
% Plot the articular spline
plot(nsplineas(:,2),nsplineas(:,1),'r-', 'LineWidth', LineWidth); hold on;
set(gca,'YDir','Reverse');
plot(splinebs(:,2), splinebs(:,1), 'b-', 'LineWidth', LineWidth)
set(gca,'YDir','Reverse');  axis equal; hold on;
plot(splineas(:,2), splineas(:,1), 'LineWidth', LineWidth)
set(gca,'YDir','Reverse');  axis equal;hold on;



%Display the warped map
figure; imagesc(wmap,[0 1]);  %caxis(gca,[0 100]); axis equal;
set(gca,'YDir','Reverse');
%title(['Spline:']);
figure; imagesc(fmap2,[0 1]);  %caxis(gca,[0 100]); axis equal;
%%

%%
figure
plot(Tland(:,1), Tland(:,2))
set(gca,'YDir','Reverse');  axis equal;
%%
oT2means-fT2means
%% 
figure;
imagesc(fmap2,[0 1]);  %caxis(gca,[0 100]); axis equal;
hold on;
plot(round(Tland(:,1)), round(Tland(:,2)), 'r', 'LineWidth', LineWidth)
% 
% 
% 
% 
% 
% 
% 
% 
%
