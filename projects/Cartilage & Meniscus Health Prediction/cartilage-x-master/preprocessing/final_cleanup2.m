% subtraction in avgt and dilation radius can be further tuned
% works for horizontal cartilage compartments
% loading in the segmentation
%load('/Users/talairach/Downloads/flatten/AFACL2005_01.mat')
load('/Users/talairach/Desktop/AFACL2008_01.mat')
no_edit_sag = double(pred_con_vol==1); % define which compartment
tba_edit_sag = no_edit_sag;

% for bookkeeping check characteristics
no_edit_props = regionprops3(no_edit_sag,{'Volume','BoundingBox','SurfaceArea'});
VOL(1) = no_edit_props.Volume;
BB(1,:) = no_edit_props.BoundingBox;
SA(1) = no_edit_props.SurfaceArea;

closed = imclose(tba_edit_sag,strel('sphere',8)); % dilate and erode with sphere
cl_sag = bwareaopen(closed,100,8); % remove weakly connected pieces

cl_props = regionprops3(cl_sag,{'Volume','BoundingBox','SurfaceArea'});
VOL(2) = cl_props.Volume;
BB(2,:) = cl_props.BoundingBox;
SA(2) = cl_props.SurfaceArea;

%%
fatskeleton = zeros(size(cl_sag));
spurred = zeros(size(cl_sag));
clean_sag = zeros(size(cl_sag));
clean_sag_s = zeros(size(cl_sag));
clean_sag_ss = zeros(size(cl_sag));

for k = 1:size(cl_sag,3)
    avgt = floor((trimmean(nonzeros(sum(cl_sag(:,:,k),1)),30)-0.25)/2);
    if isnan(avgt)
        continue
    else
    dila_test = imdilate(cl_sag(:,:,k), strel('disk',35));
    figure(2); imshow(dila_test,[])
    skel_test = bwmorph(dila_test, 'thin', Inf); 
    figure(3); imshow(skel_test,[])
    fatskeleton(:,:,k) = imdilate(skel_test,strel('disk',avgt));
    figure(4); imshow(fatskeleton(:,:,k),[])
    
    clean_sag(:,:,k) = or(fatskeleton(:,:,k),spurred(:,:,k));
    clean_sag_s(:,:,k) = bwmorph(clean_sag(:,:,k), 'spur');
    clean_sag_ss(:,:,k)=~bwmorph(~clean_sag_s(:,:,k),'spur');
    end

end

% do logical OR with fat skeleton and original

clean_filled_sag = imclose(clean_sag_ss,strel('sphere',8));
areao_test = bwareaopen(clean_filled_sag,200,4); % remove weakly connected pieces final

volumeViewer(areao_test)

slice = areao_test(:,:,78);

[B,L] = bwboundaries(slice,'noholes');
imshow(slice); hold on;
for k = 1:length(B)
boundary = B{k};
plot(boundary(:,2), boundary(:,1), 'b', 'LineWidth', 2)
end








