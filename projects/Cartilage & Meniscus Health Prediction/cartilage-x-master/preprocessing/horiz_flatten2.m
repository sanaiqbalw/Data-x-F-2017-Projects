% function horiz_flatten()
function horiz_flatten2(indices,mfc_or_lfc)
%mfc =1, lfc=2
warning('off','all')
%
load('/data/bigbone4/ciriondo/clean_all_path.mat')
load('/data/bigbone4/ciriondo/patientInfo.mat');
%
% clear all
% close all
% clc
%addpath('../PreProcessing Resources');

for j = indices
    disp(j)
    tic
    % LOAD MRI AND SEG FROM PATIENT
    split_file = strsplit(raw{j},{'/','.mat'}); fileID = split_file{end-1}
    split_name = strsplit(fileID,'_'); patientID = split_name{1}
    disp('It is good!!!');
    
    try
        
        mri = load(raw{j,1}); seg = load(raw{j,2});
        
    catch
        continue
    end
    
    
    try
        mri_cart = mri.im_store; % or some other name two options
    catch
        mri_cart = mri.full_im;
    end
    
    if size(mri_cart,3) <120
        continue
    end
    
    % LOOP THROUGH EACH HORIZONTAL COMPARTMENT (1-5) MFC - TRO
    for i = mfc_or_lfc%1:2
        
        try
            cart_comp = {'MFC','LFC','MT','LT','TRO'};
            seg_cart = double(seg.pred_con_vol==i);
            %seg_cart = double(pred_con_vol==i);
            
            % INPUT TO THIS PART IS seg_cart, mri_cart
            % STEP1: cleaning and smoothing
            seg_cart1 = bwareaopen(imclose(seg_cart,strel('sphere',5)),100,8);
            seg_cart2 = zeros(size(seg_cart1));
            
            if size(seg_cart1,3)
                
                for k = 1:size(seg_cart1,3)
                    slice_kosta = seg_cart1(:,:,k);
                    avgt = floor((trimmean(nonzeros(sum(slice_kosta,1)),30)-0.25)/2);
                    if or(isnan(avgt),length(slice_kosta(slice_kosta == 1))<150)
                        continue
                    else
                        joined = imdilate(slice_kosta, strel('disk',10));
                        CC = bwconncomp(joined);
                        cnt3=1;
                        while and(CC.NumObjects > 1,cnt3<4)
                            joined = imdilate(joined, strel('disk',20));
                            CC = bwconncomp(joined);
                            cnt3=cnt3+1;
                        end
                        fat_skeleton = imdilate(bwmorph(joined, 'thin', Inf),strel('disk',avgt));
                        clean = bwmorph(or(fat_skeleton,slice_kosta), 'spur');
                        seg_cart2(:,:,k) = ~bwmorph(~clean,'spur');
                    end
                end
                
                
                seg_cart3 = imdilate(bwareaopen(imclose(seg_cart2,strel('sphere',5)),250,4),strel('sphere',1));
                
                %% STEP2: boundaries, splines, endpoints
                k_vec = [];
                useful_splines_vec = [];
                splineas_cell = cell(1,size(seg_cart3,3)); splinebs_cell = cell(1,size(seg_cart3,3));
                for k = 1:size(seg_cart3,3)
                    %                 for k = [99:103]
                    slice_kosta = seg_cart3(:,:,k);
                    if ~any(any(slice_kosta))
                        continue
                    else
                        k_vec(length(k_vec)+1) = k;
                        useful_splines_vec(length(useful_splines_vec)+1) = 1;
                        disp(['Finding boundaries for slice ', num2str(k), ' ...']);
                        slice_kosta = imdilate(slice_kosta, strel('disk',1));
                        [end_row,end_col]= find(bwmorph(bwmorph(slice_kosta, 'thin', Inf),'endpoints'));
                        
                        [B,~] = bwboundaries(slice_kosta, 'noholes');
                        [~, min_ind(1)] = min((end_row(1)-B{1}(:,1)).^2 +(end_col(1)-B{1}(:,2)).^2);
                        [~, min_ind(2)] = min((end_row(2)-B{1}(:,1)).^2 +(end_col(2)-B{1}(:,2)).^2);
                        
                        if min_ind(2) > min_ind(1)
                            shift = -min_ind(1);
                            min_ind_shift = min_ind(2) + shift;
                        else
                            shift = size(B{1},1) - min_ind(1);
                            min_ind_shift = min_ind(2) + shift;
                        end
                        
                        boundary_shifted = circshift(B{1}, shift, 1);
                        top_spline = [boundary_shifted(end,:); boundary_shifted(1:min_ind_shift+1,:)];
                        bot_spline = boundary_shifted(min_ind_shift+1:end,:);
                        
                        % STEP3: flattening use same transformation on image and
                        % segmentation
                        % save trans!!
                        switch cart_comp{i}
                            case {'MFC','LFC','TRO'}
                                splinebs = top_spline; splineas = bot_spline;
                            case {'LT','MT'}
                                splineas = top_spline; splinebs = bot_spline;
                        end
                        splineas_cell{k} = splineas; splinebs_cell{k} = splinebs;
                        
                        %% Check if the lenghts of the splines are very different
                        splineas_len = size(splineas,1);
                        splinebs_len = size(splinebs,1);
                        len_ratio_threshold = 5; % ignore slices where the size ratio is bigger than this threshold
                        if (splineas_len/splinebs_len > len_ratio_threshold) || (splinebs_len/splineas_len > len_ratio_threshold)
                            useful_splines_vec(end) = 0;
                            continue;
                        end
                        
                        %% Check if the bounding boxes of the splines are very different
                        splineas_max = max(splineas,[],1); splineas_min = min(splineas,[],1);
                        splinebs_max = max(splinebs,[],1); splinebs_min = min(splinebs,[],1);
                        bounding_box_diff_threshold = 50; % ignore slices where the bounding box difference is bigger than this threshold
                        if sum(or(abs(splineas_max-splinebs_max)>bounding_box_diff_threshold, abs(splineas_max-splinebs_max)>bounding_box_diff_threshold)) > 0
                            useful_splines_vec(end) = 0;
                            continue;
                        end
                    end
                end
                %                 figure
                %                 stem(k_vec, useful_splines_vec, 'LineWidth', 1.5);
                %                 grid on;
                %                 k_vec;
                
                %% For useful_splines=0 take the closest useful_splines=1 splines
                k_vec_new = k_vec;
                for ind = 1:length(k_vec)
                    if useful_splines_vec(ind) == 0
                        flag_found = 0;
                        delta = 1;
                        while flag_found == 0
                            if abs(delta) > length(k_vec)
                                ind_new = ind;
                                disp('Could not find any neighbor with good splines, keeping the same spline!!!');
                                continue;
                            end
                            ind_new = ind + delta;
                            if (ind_new > 0) && (ind_new < length(k_vec)+1)
                                if useful_splines_vec(ind_new) == 1
                                    k_vec_new(ind) = k_vec(ind_new);
                                    flag_found = 1;
                                end
                            end
                            if delta > 0
                                delta = -delta;
                            else
                                delta = -delta+1;
                            end
                        end
                        %                         disp(['ind = ', num2str(ind), ', ind_new = ', num2str(ind_new)]);
                        splineas_cell(k_vec(ind)) = splineas_cell(k_vec(ind_new));
                        splinebs_cell(k_vec(ind)) = splinebs_cell(k_vec(ind_new));
                    end
                end
                %                 figure
                %                 plot(k_vec, k_vec_new, 'o-', 'LineWidth', 1.5);
                %                 grid on;
                
                %% Run the flattening using the new splines
                seg_flat = zeros(size(seg_cart1));
                im_flat = zeros(size(seg_cart1));
                coord_save = cell(1,size(seg_cart1,3));
                BBs = zeros(length(k_vec), 4); BBs2 = zeros(length(k_vec), 4);
                cnt=1;
                for k = k_vec
                    disp(['Flattening slice ', num2str(k), ' ...']);
                    imslice = mri_cart(:,:,k);
                    splineas = splineas_cell{k}; splinebs = splinebs_cell{k};
                    [im_flat(:,:,k), seg_flat(:,:,k),coord_save{k}] = ...
                        flat_2D_cartilage_warping_claudia(imslice, splineas, splinebs, 0);
                    
                    %                     slice_kosta = seg_cart3(:,:,k);
                    %                     d = 15;
                    %                     spline_full = [splinebs; splineas];
                    %                     axis_vec = [min(spline_full(:,2))-d max(spline_full(:,2))+d min(spline_full(:,1))-d max(spline_full(:,1))+d];
                    %                     LineWidth = 1.5;
                    %                     fig_size = [0.66 1];
                    %                     plot_rows = 2; plot_cols = 2;
                    %                     figure
                    %                     set(gcf,'units','normalized','outerposition',[0.5-fig_size/2 fig_size]);
                    %                     subplot(plot_rows, plot_cols, 1);
                    %                     show_image(imslice, 'image');
                    %                     hold on;
                    %                     plot(splinebs(:,2), splinebs(:,1), 'bo-', splineas(:,2), splineas(:,1), 'ro-', 'Linewidth', 1.5);
                    %                     title(['image, slice k=', num2str(k)]);
                    %                     legend('splinebs','splineas'); %axis(axis_vec);
                    %                     subplot(plot_rows, plot_cols, 2);
                    %                     show_image(slice_kosta, 'image');
                    %                     hold on;
                    %                     plot(splinebs(:,2), splinebs(:,1), 'bo-', splineas(:,2), splineas(:,1), 'ro-', 'Linewidth', 1.5);
                    %                     title(['segmenation, slice k=', num2str(k)]);
                    %                     legend('splinebs','splineas'); %axis(axis_vec);
                    %                     subplot(plot_rows, plot_cols, 3);
                    %                     show_image(im_flat(:,:,k), 'image');
                    %                     subplot(plot_rows, plot_cols, 4);
                    %                     show_image(seg_flat(:,:,k), 'image');
                    
                    prop = regionprops(seg_flat(:,:,k),'BoundingBox');
                    BBs(cnt,:) = prop.BoundingBox;
                    cnt = cnt+1;
                end
                
                BBs(:,1) = min(BBs(:,1));
                BBs(:,3) = max(BBs(:,3));
                BBs(:,4) = max(BBs(:,4));
                BBs = floor(BBs);
                BBs = BBs+[-4,-2,8,4];
                cnt2 = 1;
                
                for k = 1:size(seg_flat,3)
                    slice = seg_flat(:,:,k);
                    if ~any(any(slice))
                        continue
                    else
                        imslice = im_flat(:,:,k);
                        im_flat_crop(:,:,cnt2) = imcrop(imslice, BBs(cnt2,:));
                        seg_flat_crop(:,:,cnt2) = imcrop(slice,BBs(cnt2,:));
                        cnt2=cnt2+1;
                    end
                end
                %
                %                 %--------------
                %                 % OUTPUT IS FLATTENED SEGMENTATION, MRI, COORDINATES
                %
                %                 %% ARRANGING AND SAVING FILES
                switch i % switch is hardcoded, more elegant implementation would
                    % include regex on first row of raw, titles
                    case 1
                        WORMS = raw{j,3}; BME = raw{j,5};
                    case 2
                        WORMS = raw{j,4}; BME = raw{j,6};
                end
                
                if patientInfo.(patientID).GENDER == 'F'
                    GENDER = 1;
                else
                    GENDER = 0;
                end
                AGE = patientInfo.(patientID).AGE;
                BMI = patientInfo.(patientID).BMI;
                [H,W,D] = size(im_flat_crop);
                
                % [WORMS BME GENDER AGE BMI H W D _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ]
                mri_array = reshape(im_flat_crop,[1,H*W*D]);
                seg_array = reshape(seg_flat_crop,[1,H*W*D]);
                
                mri_save = horzcat(WORMS,BME,GENDER,AGE,BMI,H,W,D,mri_array);
                seg_save = horzcat(WORMS,BME,GENDER,AGE,BMI,H,W,D,seg_array);
                %
                % SAVING AS A .RAW FILE
                cd(['/data/bigbone4/DeepLearning_temp/Data/CartilageX/mri/flat_',cart_comp{i},'/'])
                fid1 = fopen([fileID,'.raw'],'w+');
                fwrite(fid1,mri_save,'single'); fclose(fid1);
                
                cd(['/data/bigbone4/DeepLearning_temp/Data/CartilageX/seg/flat_',cart_comp{i},'/'])
                fid2 = fopen([fileID,'.raw'],'w+');
                fwrite(fid2,seg_save,'single'); fclose(fid2);
                
                cd(['/data/bigbone4/DeepLearning_temp/Data/CartilageX/morph/coords_',cart_comp{i},'/'])
                save([fileID,'.mat'],'coord_save')
                
                fprintf('Patient %s : %s processed.\n',fileID,cart_comp{i});
                toc
                
                clearvars -except seg mri_cart raw patientID fileID patientInfo j i mfc_or_lfc
                
                
            end
        catch e
            fprintf(1,'The identifier was:\n%s',e.identifier);
            fprintf(1,'There was an error! The message was:\n%s',e.message);
            fprintf('\n\n\n FileID %s %s failed... \n\n IT FAILED!! \n',fileID,cart_comp{i})
            clearvars -except seg mri_cart raw patientID fileID patientInfo j i mfc_or_lfc
            continue
        end
        
    end
    clearvars -except patientInfo raw j mfc_or_lfc
    disp(['Elapsed time = ', num2str(toc), ' seconds'])
end
return

function [image_flattened, seg_flattened, coord_save] = flat_2D_cartilage_warping_claudia(image_to_flatten, splineas, splinebs, plot_flag)

% show_type = 'imshow';
show_type = 'image';

if plot_flag == 1
    figure
    show_image(image_to_flatten, show_type);
    hold on;
    plot(splinebs(:,2), splinebs(:,1), 'b', splineas(:,2), splineas(:,1), 'r');
    legend('splinebs','splineas');
end
%%

grid = zeros(size(image_to_flatten));

splinebs=round(splinebs);
splineas=round(splineas);

for i = 1:length(splinebs)
    grid(splinebs(i,1),splinebs(i,2)) = 1;
end

for i = 1:length(splineas)
    grid(splineas(i,1), splineas(i,2)) = 2;
end

if plot_flag == 1
    figure
    imshow(grid)
end
%%
% Grid is ready to be used as an input into SolveLaplaceEquation2D.m code.

potentialMap = SolveLaplaceEquation2D(grid,0);

if plot_flag == 1
    figure
    imshow(potentialMap)
end
%%
spline_mask = grid;
spline_mask(spline_mask>0) = 1;
spline_mask = imfill(spline_mask,'holes');

if plot_flag == 1
    figure
    imagesc(potentialMap.*spline_mask)
end
%%
% Laplacian equation filled in the values between two spline surfaces.
%
% Next steps: 1. Get cartilage contour 2. Get first endpoint 3. Get second
% endpoint 4. Get bone border 5. Get articular border 6. Make them share endpoints


%splinebs = [smooth(splinebs(:,1),7) smooth(splinebs(:,2),7)];
%splineas = [smooth(splineas(:,1),7) smooth(splineas(:,2),7)];
if plot_flag == 1
    figure
    plot(splinebs(:,2), splinebs(:,1))
    set(gca,'YDir','Reverse');  axis equal; hold on;
    plot(splineas(:,2), splineas(:,1))
    set(gca,'YDir','Reverse');  axis equal;
end
%%
% "Get matchings between the bone and articular surfaces", bone surface
% chosen as a reference since edema/other findings will be taken from there
%
% This calls on function getstreamlines2D.m, which matches bone to articular
% points based on a potential map

streamlines = getstreamlines2D_kosta(potentialMap,splinebs,2,spline_mask,0);
%     streamlines = getstreamlines2D(potentialMap,splinebs,2,spline_mask,0);
%     streamlines = getstreamlines2D(potentialMap,splinebs,2,spline_mask,1);
%     streamlines = getstreamlines2Dnew(potentialMap,splinebs,2,spline_mask,0);
%%
% Potential starting pointStreamlines from potential map! Get the number
% of streamlines, prepare variables for new splines and thickness value.

nstreamlines = length(streamlines);
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
matchcd = matchpointsinC(cdist,cumdistb);
%matchcd = matchpointsinC(cdist,cumdistb);
cland = nsplinebs(matchcd,:);
% Get the number of landmarks in the bone cartilage interface
nlands = length(cland);
%%
% Flatten the cartilage wrt the first spline point in b (fptb), establishes
% target landmarks for bone interface, on a horizontal line at the same "y" than
% that of fptb

% Tlands_direction calculated in other situations, will be assinged a 1 for
% femoral compartments and -1 for tibia compartments
%%
fptb = nsplinebs(1,:)+[-55 0];
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
%Tlanda = round(Tlanda);
%%
% Get Fornefettcoefficients and applying them, this is used to do backward
% mapping,
%

[alphasv,av] = getFornefettcoefficients(cland,Tland,[]);

[wsplineb] = applyForneffetcoeffs(nsplinebs,cland,alphasv,av);
[wsplinea] = applyForneffetcoeffs(nsplineas,cland,alphasv,av);

[alphas,a] = getFornefettcoefficients(Tland,cland,[]);

maxx = round(max(Tland(:,1)))+10;
maxx=512;
maxy = round(max(Tland(:,2)))+10;
maxy=512;
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
% % img = load('/Users/talairach/Desktop/ML_examples/mri_45.mat');
% % nmap = img.im_store(100:400,100:500,22);
% load(['D:\Google Drive\Berkeley\Courses\Data X IEOR 290\Project\Cartilage-X\PreProcessing Resources\manualExamplesforKosta\mri_', num2str(man_image_num), '.mat']);
% % nmap = img.im_store(100:400, 100:500, num_start);
% nmap = im_store(:, :, num_start);
nmap = image_to_flatten;
if plot_flag == 1
    figure
    imshow(nmap);
end

[otx,oty] = meshgrid(1:size(image_to_flatten,2),1:size(image_to_flatten,1));
fmap = interp2(otx,oty,nmap,reshape(trc(:,2),maxx,maxy),reshape(trc(:,1),maxx,maxy),'spline');
%fmap = interp2(nmap,reshape(trc(:,2),maxx,maxy),reshape(trc(:,1),maxx,maxy),'spline');
fmap = fmap';
if plot_flag == 1
    imagesc(fmap,[0 1]);
end
%%
% Trying to apply coeff

[alphas2,a2] = getFornefettcoefficients(Tland,cland,[]);
maxx2 = round(max(Tland(:,1)))+100;
maxx2 = 512;
maxy2 = round(max(Tland(:,2)))+100;
maxy2 = 512;
[tx2, ty2] = ndgrid(1:maxx2,1:maxy2);
tx2 = tx2(:);
ty2 = ty2(:);
[trc2] = applyForneffetcoeffs([tx2 ty2],Tland,alphas2,a2);
[otx2,oty2] = meshgrid(1:size(image_to_flatten,2),1:size(image_to_flatten,1));
fmap2 = interp2(otx2,oty2,nmap,reshape(trc2(:,2),maxx2,maxy2),reshape(trc2(:,1),maxx2,maxy2),'spline');
fmap2 = fmap2';
if plot_flag == 1
    imagesc(fmap2,[0 1]);
end
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

% cmask = poly2mask([nsplinebs(:,1); flipdim(nsplineas(2:end-1,1),1)],[nsplinebs(:,2); flipdim(nsplineas(2:end-1,2),1)],size(testmask,1),size(testmask,2));
cmask = poly2mask([nsplinebs(:,1); flipdim(nsplineas(2:end-1,1),1)],[nsplinebs(:,2); flipdim(nsplineas(2:end-1,2),1)],size(image_to_flatten,1),size(image_to_flatten,2));
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


if plot_flag == 1
    figure
    imagesc(fmap,[0 1])
    figure
    imagesc(flipdim(wmap,1),[0 1])
    set(gca,'XDir','Reverse');  axis equal;
end
%
% Plot the bone spline
if plot_flag == 1
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
    
    
    LineWidth = 2;
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
    %     oT2means-fT2means
    %%
    figure;
    imagesc(fmap2,[0 1]);  %caxis(gca,[0 100]); axis equal;
    hold on;
    plot(round(Tland(:,1)), round(Tland(:,2)), 'r', 'LineWidth', LineWidth)
end

coord_save = {[tx2 ty2],Tland,alphas2,a2};
image_flattened = fmap2;
seg_flattened = wmask;
return



function Streamlines = getstreamlines2D_kosta(PotentialMap,bspline,aval,mask,displayflag)
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
    nsteps = 1; % number of while iterations
    while isarticular==0
        if (nsteps > 1000)
            disp('Reached 1000 steps, terminating while loop');
            Streamlines{npt} = [];
            break
        end
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
            %             disp(['steamline found in ', num2str(nsteps), ' steps']);
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
        nsteps = nsteps + 1;
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
return

