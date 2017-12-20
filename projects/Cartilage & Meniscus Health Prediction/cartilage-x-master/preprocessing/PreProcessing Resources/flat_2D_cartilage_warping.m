function [fIma fmyinfotosave MeanT2OrigFlat StdT2OrigFlat Thicks] = flat_2D_cartilage_warping(Ima,myinfotosave,matching_method,noffsets,isboneref,T2Th,displayflag)
% [fIma fmyinfotosave MeanT2OrigFlat StdT2OrigFlat Thicks] = flat_2D_cartilage_warping(Ima,myinfotosave,noffsets,isboneref,T2Th,displayflag)
%  
% Function for flattening of cartilage segmentations done with IPP. The function also flattens the corresponding images.
% 
% Inputs:
% Ima             - 3D array of doubles. The 3D image (e.g. T2 map). The slices are in the 3rd dimension.
% myinfotosave    - The IPP cartilage segmentation structure.
% noffsets        - Since cartilage flattening is predominantly done for texture analysis using gray-level co-occurrence matrices (GLCM), this is a scalar that tells the program the number of offsets to be used in GLCM.
%                   The purpose of noffsets is for cases where the cartilage is split into multiple sements in a single slice so we can separate the flattened images accordingly.
% isboneref       - isboneref=1 The bone-cartilage interface is the reference for flattening, i.e. the bone-cartilage interface will look like a straight line.
%                   isboneref=0 The articular surface is the reference for flattening, i.e. the articular surface will look like a straight line.
% T2Th            - It is a scalar indicating that all Ima values greater than T2Th should be cropped to T2Th.
%                   T2Th=[] does not crop Ima.
% displayflag     - displayflag=1 Displays partial results.
%                   displayflag=0 Displays nothing.
%
% Outputs:
% fIma            - Cell of size nslicesx1 where nslices=size(Ima,3). Each cell is a flattened slice of Ima.
% fmyinfotosave   - A new IPP cartilage segmentation structure with the flattened cartilage segmentations.
% MeanT2OrigFlat  - 2D array of doubles. Each row has the mean value of an original region of interest (ROI) and the mean value of the corresponding flattened ROI.
% StdT2OrigFlat   - 2D array of doubles. Each row has the std value of an original ROI and the std of the corresponding flattened ROI.
% Thicks          - A column vector of doubles with the cartilage thickness of every single point.
%
%
%
% by
% Julio Carballido-Gamio
% 2007-2009
% Julio.Carballido@gmail.com
%

% VersionString = '1.01.01'  ;                 % major, minor and source version
% VersionTrack(mfilename,VersionString);       % This call stores the version string for inclusion in outputs
%cnlands = 50;
displaywsplines = 0;
% matching_method = 3; % 1 - Minimum euclidean distances; 2 - Normal vectors; 3 - Laplacian equation
displayPotentialMap = 0;
displayStreamLines = 0;
displayStreamlinesonMap = 0;
rstreamlines = 1; % Flag to calculate the 2D thickness based on the streamlines (rstreamlines=1) or based on the Euclidean distance between matched points (rstreamlines=0); only for matching_method=3
%-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
if ~exist('Ima','var'),                 fIma = []; fmyinfotosave = []; return;          end
if ~exist('myinfotosave','var'),        fIma = []; fmyinfotosave = []; return;          end
if ~exist('noffsets','var'),            noffsets = 5;                                   end
if ~exist('isboneref','var'),           isboneref = 1;                                  end
if ~exist('T2Th','var'),                T2Th = [];                                      end
if ~exist('displayflag','var'),         displayflag = 0;                                end

%-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
% Crop Ima
if ~isempty(T2Th)
    Ima(Ima>T2Th) = T2Th+1;
end
%-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
% Check S3 vs IPP
if ~isfield(myinfotosave,'IPPversion') % Means it was created with S3 so the y coordinates are inverted
    for nslice=1:myinfotosave.w3
        for nspline=1:myinfotosave.dataperslice{nslice}.nsplines
            myinfotosave.dataperslice{nslice}.ycoordinatesspl{nspline} = myinfotosave.w1-myinfotosave.dataperslice{nslice}.ycoordinatesspl{nspline}+1;
        end
    end
end
%-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
% Create a new IPP structure
fmyinfotosave = myinfotosave;
% Save the bone flag
fmyinfotosave.isboneref_flattened = isboneref;
% Clear fmyinfotosave 
for nslice=1:myinfotosave.w3
    % Clean closed splines
    if isfield(myinfotosave.dataperslice{1},'ncsplines')
        fmyinfotosave.dataperslice{nslice}.ncsplines = 0;
        fmyinfotosave.dataperslice{nslice}.xcoordinatescspl = cell(1,1);
        fmyinfotosave.dataperslice{nslice}.ycoordinatescspl = cell(1,1);
    end
    % Clean boxes
    if isfield(myinfotosave.dataperslice{1},'nboxes')
        fmyinfotosave.dataperslice{nslice}.nboxes = 0;
        fmyinfotosave.dataperslice{nslice}.xcoordinatesb = cell(1,1);
        fmyinfotosave.dataperslice{nslice}.ycoordinatesb = cell(1,1);
    end
    % Clean circles
    if isfield(myinfotosave.dataperslice{1},'ncircles')
        fmyinfotosave.dataperslice{nslice}.ncircles = 0;
        fmyinfotosave.dataperslice{nslice}.xcentres = cell(1,1);
        fmyinfotosave.dataperslice{nslice}.ycentres = cell(1,1);
        fmyinfotosave.dataperslice{nslice}.radii = cell(1,1);
    end
    % Clean masks
    if isfield(myinfotosave.dataperslice{1},'nmasks')
        fmyinfotosave.dataperslice{nslice}.nmasks = 0;
        fmyinfotosave.dataperslice{nslice}.xcoordinates = cell(1,1);
        fmyinfotosave.dataperslice{nslice}.ycoordinates = cell(1,1);
    end
end
if isfield(myinfotosave,'regGL')
    fmyinfotosave.regGL = struct('RAF',[], ...         % This field is to know the type of registration
                                'Metric', ...         % This field is to know the metric used for optimization
                                'nscales', ...        % This field is to know the number of scales in a multi-resolution approach
                                'T',[], ...           % This field is for the affine transformation parameters
                                'sizeR',[], ...       % This field is for the matrix size of the reference image
                                'resR',[], ...        % This field is for the spatial resolution of the reference image
                                'filenameR',[], ...   % This field is for the filename of the reference image 
                                'toplcR',[], ...      % This field is for the toplc coordinate of the reference image
                                'cosinesR',[], ...    % This field is for the cosines (patient coordinates) of the reference image
                                'nboxesR',0,'xcoordinatesbR',cell(1,1),'ycoordinatesbR',cell(1,1), ... % For box ROIs
                                'Regularization',[], ...  % This field is to know the regularization used in FFD
                                'z1',[],'zend',[], ... % Range of slices for non-linear registration on a slice-by-slice basis
                                'Spacings',cell(1,1), ...          % This field is for the spacings between control points in the final control points mesh
                                'ControlPointsMesh',cell(1,1), ... % This field is for the final control points mesh
                                'Tr',cell(1,1), ... % For the final row-displacement fields
                                'Tc',cell(1,1), ... % For the final column-displacement fields
                                'Tz',cell(1,1));    % For the final z-displacement fields
end
%-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
% Check main directionality of cartilage contours based on PCA
maindirs = [];
fpts = [];
lpts = [];
mainvalue2=[];







for nslice=1:myinfotosave.w3
    for nspline=1:2:myinfotosave.dataperslice{nslice}.nsplines
        [eig_vecs, eig_value,eig_cenid] = get_PCA([[myinfotosave.dataperslice{nslice}.xcoordinatesspl{nspline}; myinfotosave.dataperslice{nslice}.xcoordinatesspl{nspline+1}], ...
                            [myinfotosave.dataperslice{nslice}.ycoordinatesspl{nspline}; myinfotosave.dataperslice{nslice}.ycoordinatesspl{nspline+1}]]);
        maindirs = [maindirs eig_vecs(:,1)];
        mainvalue2=[mainvalue2 eig_value(2)];
        %fpts = [fpts; [myinfotosave.dataperslice{nslice}.xcoordinatesspl{nspline}(1)  myinfotosave.dataperslice{nslice}.ycoordinatesspl{nspline}(1)]];
        %lpts = [lpts; [myinfotosave.dataperslice{nslice}.xcoordinatesspl{nspline}(end)  myinfotosave.dataperslice{nslice}.ycoordinatesspl{nspline}(end)]];
    end
end
maindirs = mean(maindirs,2);
meanvalue2= mean(mainvalue2);
maindir = atan2(maindirs(2),maindirs(1));
maindir(maindir<0) = maindir(maindir<0)+pi;
maindir = maindir*180/pi;
Disp(strcat('meanvalue2=', num2str(meanvalue2)));
Disp(strcat('maindir=', num2str(maindir)));

mainorientation = 'Horizontal';
%Disp('maindir=',maindir);
if maindir>45 & maindir<125
    mainorientation = 'Vertical';
end

diary  off;
global MYINFO;
MYINFO.RecorderFlattenOn=false;
%[path_MAT1 rootname_MAT1 ext_MAT1]=fileparts(myinfotosave.matfiles);
if isequal(exist(myinfotosave.pathname, 'dir'),7)
 log_file=strcat(myinfotosave.pathname,'Flattening_Textrue_logfile.txt');
else
    log_file='/data/bigbone5/IppExamples/IppProcessingRecordings/Flattening_Textrue_logfile.txt';
end


% Check where the first point is
if strcmp(mainorientation,'Horizontal')
    %% Cartilage info and Intersection detection added by zhihong
  for nslice=1:myinfotosave.w3 
    % for Opened Splines
    if myinfotosave.dataperslice{nslice}.nsplines > 0
        % if roi matrix is in wrong direction
        xcoord=myinfotosave.dataperslice{nslice}.xcoordinatesspl{1};
        % check the first and last points of the top spline to see whether the line was drawn A->P or
        % P-> A 
        if (xcoord(1) < xcoord(end) ) && (maindir <45  || maindir>170) && meanvalue2<0.04     %% for LT and MT   
            %MYINFO.RecordOn=1;   %start to record the erroo message
            %Record_IPP( )
            
            diary (log_file);
             
            MYINFO.RecorderFlattenOn=true;
            %Disp(['Mat file:', strcat(rootname_MAT1, ext_MAT1)]);
            Disp(['Slice No: ', num2str(nslice)]); 
            Disp('a slice ROI with rotated matrix is detected and corrected');
            for nspline=1:myinfotosave.dataperslice{nslice}.nsplines
                % clean the temp variable to swap
                pointstempx=[];
                pointstempy=[];
                % flip the cooridnates and save to temp variable
                pointstempx = flipud(myinfotosave.dataperslice{nslice}.xcoordinatesspl{nspline});
                pointstempy = flipud(myinfotosave.dataperslice{nslice}.ycoordinatesspl{nspline});
                % clear the spline var.
                myinfotosave.dataperslice{nslice}.xcoordinatesspl{nspline} = [];             
                myinfotosave.dataperslice{nslice}.ycoordinatesspl{nspline} = []; 
                % move the data from temp var to spline var.            
                myinfotosave.dataperslice{nslice}.xcoordinatesspl{nspline} = pointstempx; 
                myinfotosave.dataperslice{nslice}.ycoordinatesspl{nspline} = pointstempy; 
            end
         elseif ( xcoord(1) > xcoord(end) ) && maindir >125 && meanvalue2>0.04      %% for MFC and LFC
             
            diary (log_file);
             
            MYINFO.RecorderFlattenOn=true;
            %Disp(['Mat file:', strcat(rootname_MAT1, ext_MAT1)]);
            Disp(['Slice No: ', num2str(nslice)]); 
            Disp('a slice ROI with rotated matrix is detected and corrected');
            for nspline=1:myinfotosave.dataperslice{nslice}.nsplines
                % clean the temp variable to swap
                pointstempx=[];
                pointstempy=[];
                % flip the cooridnates and save to temp variable
                pointstempx = flipud(myinfotosave.dataperslice{nslice}.xcoordinatesspl{nspline});
                pointstempy = flipud(myinfotosave.dataperslice{nslice}.ycoordinatesspl{nspline});
                % clear the spline var.
                myinfotosave.dataperslice{nslice}.xcoordinatesspl{nspline} = [];             
                myinfotosave.dataperslice{nslice}.ycoordinatesspl{nspline} = []; 
                % move the data from temp var to spline var.            
                myinfotosave.dataperslice{nslice}.xcoordinatesspl{nspline} = pointstempx; 
                myinfotosave.dataperslice{nslice}.ycoordinatesspl{nspline} = pointstempy; 
             end 
              %MYINFO.RecordOn=0;   % end to record
              %diary off;
        end
    end
  end
   % ended by zhihong
   
for nslice=1:myinfotosave.w3
    for nspline=1:2:myinfotosave.dataperslice{nslice}.nsplines
        %eig_vecs = get_PCA([[myinfotosave.dataperslice{nslice}.xcoordinatesspl{nspline}; myinfotosave.dataperslice{nslice}.xcoordinatesspl{nspline+1}], ...
                            %[myinfotosave.dataperslice{nslice}.ycoordinatesspl{nspline}; myinfotosave.dataperslice{nslice}.ycoordinatesspl{nspline+1}]]);
        %maindirs = [maindirs eig_vecs(:,1)];
        fpts = [fpts; [myinfotosave.dataperslice{nslice}.xcoordinatesspl{nspline}(1)  myinfotosave.dataperslice{nslice}.ycoordinatesspl{nspline}(1)]];
        lpts = [lpts; [myinfotosave.dataperslice{nslice}.xcoordinatesspl{nspline}(end)  myinfotosave.dataperslice{nslice}.ycoordinatesspl{nspline}(end)]];
    end
end
    % Compare x-coordinates
    diffx = mean(lpts(:,1)-fpts(:,1));
    % Should be positive for femoral compartments, and negative for tibia compartments
    if diffx>0
        Tlands_direction = 1;
    else
        Tlands_direction = -1;
    end
else % Mean Vertical
     %% Cartilage info and Intersection detection added by zhihong
  for nslice=1:myinfotosave.w3 
    % for Opened Splines
    if myinfotosave.dataperslice{nslice}.nsplines > 0
        % if roi matrix is in wrong direction
        ycoord=myinfotosave.dataperslice{nslice}.ycoordinatesspl{1};
        % check the first and last points of the top spline to see whether the line was drawn A->P or
        % P-> A 
        if (ycoord(1) < ycoord(end) ) &&      maindir >=75      %% for pattela 
            %MYINFO.RecordOn=1;  % start to recording
            %Record_IPP( );
             
            diary (log_file);
             
            MYINFO.RecorderFlattenOn=true;
            %Disp(['Mat file:', strcat(rootname_MAT1, ext_MAT1)]);
            Disp(['Slice No: ', num2str(nslice)]); 
            Disp('a slice ROI with rotated matrix is detected and corrected');
            for nspline=1:myinfotosave.dataperslice{nslice}.nsplines
                % clean the temp variable to swap
                pointstempx=[];
                pointstempy=[];
                % flip the cooridnates and save to temp variable
                pointstempx = flipud(myinfotosave.dataperslice{nslice}.xcoordinatesspl{nspline});
                pointstempy = flipud(myinfotosave.dataperslice{nslice}.ycoordinatesspl{nspline});
                % clear the spline var.
                myinfotosave.dataperslice{nslice}.xcoordinatesspl{nspline} = [];             
                myinfotosave.dataperslice{nslice}.ycoordinatesspl{nspline} = []; 
                % move the data from temp var to spline var.            
                myinfotosave.dataperslice{nslice}.xcoordinatesspl{nspline} = pointstempx; 
                myinfotosave.dataperslice{nslice}.ycoordinatesspl{nspline} = pointstempy; 
            end
        elseif (ycoord(1) > ycoord(end) ) &&      maindir <75      %% for TRO
             
            diary (log_file);
             
            MYINFO.RecorderFlattenOn=true;
            %Disp(['Mat file:', strcat(rootname_MAT1, ext_MAT1)]);
            Disp(['Slice No: ', num2str(nslice)]); 
            Disp('a slice ROI with rotated matrix is detected and corrected');
            for nspline=1:myinfotosave.dataperslice{nslice}.nsplines
                % clean the temp variable to swap
                pointstempx=[];
                pointstempy=[];
                % flip the cooridnates and save to temp variable
                pointstempx = flipud(myinfotosave.dataperslice{nslice}.xcoordinatesspl{nspline});
                pointstempy = flipud(myinfotosave.dataperslice{nslice}.ycoordinatesspl{nspline});
                % clear the spline var.
                myinfotosave.dataperslice{nslice}.xcoordinatesspl{nspline} = [];             
                myinfotosave.dataperslice{nslice}.ycoordinatesspl{nspline} = []; 
                % move the data from temp var to spline var.            
                myinfotosave.dataperslice{nslice}.xcoordinatesspl{nspline} = pointstempx; 
                myinfotosave.dataperslice{nslice}.ycoordinatesspl{nspline} = pointstempy; 
            end
            %diary off;
            %MYINFO.RecordOn=0 ;       % end to recording 
        end
    end
  end
  diary  off;
  %% ended by zhihong
  
for nslice=1:myinfotosave.w3
    for nspline=1:2:myinfotosave.dataperslice{nslice}.nsplines
        %eig_vecs = get_PCA([[myinfotosave.dataperslice{nslice}.xcoordinatesspl{nspline}; myinfotosave.dataperslice{nslice}.xcoordinatesspl{nspline+1}], ...
                            %[myinfotosave.dataperslice{nslice}.ycoordinatesspl{nspline}; myinfotosave.dataperslice{nslice}.ycoordinatesspl{nspline+1}]]);
        %maindirs = [maindirs eig_vecs(:,1)];
        fpts = [fpts; [myinfotosave.dataperslice{nslice}.xcoordinatesspl{nspline}(1)  myinfotosave.dataperslice{nslice}.ycoordinatesspl{nspline}(1)]];
        lpts = [lpts; [myinfotosave.dataperslice{nslice}.xcoordinatesspl{nspline}(end)  myinfotosave.dataperslice{nslice}.ycoordinatesspl{nspline}(end)]];
    end
end
    % Compare y-coordinates
    diffy = mean(lpts(:,2)-fpts(:,2));
    % Should be positive for patella, and negative for trochlea
    if diffy<0 % Means patella
        Tlands_direction = -1;
    else % Means trochlea
        Tlands_direction = 1;
    end
end
%-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
% Save the flattening method in the IPP structure
switch matching_method
    case 1
        fmyinfotosave.flattening_method_2D = 'Minimum Euclidean distances';
    case 2
        fmyinfotosave.flattening_method_2D = 'Normal vectors';
    case 3
        fmyinfotosave.flattening_method_2D = 'Laplace''s Equation';
end
%-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
% fIma will have the final flattened cartilage on a slice by slice basis
fIma = cell(myinfotosave.w3,1); 
%-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
% The following 2 variables are to compare T2 global means of original and flattened cartilages
oT2means = [];
fT2means = [];
% The following 2 variables are to compare T2 global stds of original and flattened cartilages
oT2stds = [];
fT2stds = [];
% The following variable is to calculate the mean cartilage thickness
Thicks = []; 
% Go slice by slice
for nslice=1:myinfotosave.w3  
    %-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    % Check if the slice has any segmented cartilage
    if myinfotosave.dataperslice{nslice}.iscartilagesegmented==0
        continue; % Go to the next loop
    end    
    %-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    nmap = Ima(:,:,nslice);
    %-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    % sliceMaps is a temporal var to save the flattened maps for the current slice
    sliceMaps = [];
    % Go cartilage segment by cartilage segment
    for nspline=1:2:myinfotosave.dataperslice{nslice}.nsplines
        % Copy the number of landmarks
        %nlands = cnlands;
        %-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        % Get the control points of the splines of the bone-cartilage interface
        xptsb = myinfotosave.dataperslice{nslice}.xcoordinatesspl{nspline};
        yptsb = myinfotosave.dataperslice{nslice}.ycoordinatesspl{nspline};    
        % Get the spline of the bone cartilage interface. Try to have a number of points between two control points that is proportional to the length of the curve connecting them 
        splineb = bezieruniformsampling([xptsb yptsb zeros(length(xptsb),1)],myinfotosave.deltax,0.001); % We are assuming square pixels, which is common
        splineb(:,3) = [];      
        % Get the control points of the spline of the articular surface
        xptsa = myinfotosave.dataperslice{nslice}.xcoordinatesspl{nspline+1};
        yptsa = myinfotosave.dataperslice{nslice}.ycoordinatesspl{nspline+1};
        % Get the spline of the articular surface. Try to have a number of points between two control points that is proportional to the length of the curve connecting them 
        splinea = bezieruniformsampling([xptsa yptsa zeros(length(xptsa),1)],myinfotosave.deltax,0.001);
        splinea(:,3) = [];  
        %-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        % Get the cartilage mask
        cmask = poly2mask([splineb(:,1); flipdim(splinea(2:end-1,1),1)],[splineb(:,2); flipdim(splinea(2:end-1,2),1)],myinfotosave.w1,myinfotosave.w2);
        % Mask the map
        cmap = cmask.*nmap;
        % Get non-flattened mean and std
        validpts = cmask>0;
        oT2means = [oT2means; mean(cmap(validpts))];
        oT2stds = [oT2stds; std(cmap(validpts))];
        %-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        % Swap the splines if necessary
        if isboneref==0         
            temp_splb= splineb;
            splineb = splinea;
            splinea = temp_splb;            
        end
        %-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        % Match the bone-cartilage interface and the articular surface
        switch matching_method
            case 1 % Match the splines based on minimum Euclidean distances
                % Matching and distances
                [matchings r] = matchpointsinC(splineb,splinea(2:end-1,:));
                matchings = matchings+1;
                splinea = splinea(matchings,:);  
                % Get directionality
                theta = atan2(splinea(:,2)-splineb(:,2),splinea(:,1)-splineb(:,1));
                theta(theta<0) = 2*pi+theta(theta<0);
            case 2 % Match the splines using normal vectors to the spline used as reference
                mydeltaspl = 10;
                nptssplineb = size(splineb,1);
                myvector = 1:mydeltaspl:nptssplineb; 
                nprofiles = length(myvector);
                [theta] = get_normals(splineb,1);
                if isboneref==0
                    theta = theta+pi;
                    pos2pi = find(theta>=2*pi);
                    if ~isempty(pos2pi)
                        theta(pos2pi) = theta(pos2pi)-2*pi;
                    end
                end
                splineb = splineb(myvector,:);
                theta = [theta; theta(end)];        
                theta = theta(myvector);       
                % "r" is a threshold that sets how far to trace the line profiles. It is given in pixels
                r = 30;
                posmatch = zeros(nprofiles,1);
                for npt=1:nprofiles
                    vnormalsx = linspace(splineb(npt,1),splineb(npt,1)+r*cos(theta(npt)),r);
                    vnormalsy = linspace(splineb(npt,2),splineb(npt,2)+r*sin(theta(npt)),r); 
                    [mattchings dists] = matchpointsinC(splinea,[vnormalsx' vnormalsy']);
                    [mindist posmatch(npt)] = min(dists);
                end
                splinea = splinea(posmatch,:);  
                % Get the distance between the two surfaces 
                r = sqrt(sum((splinea-splineb).^2,2));
            case 3 % Laplace's equation
                % Reduce the working area to improve speed
                % We are assuming here that ROI voxels will not be right on the edge of the image, otherwise getStreamlines2D could fail by going to a row or column coordinate "0"
                minx = floor(min([splinea(:,1); splineb(:,1)]));
                maxx = ceil(max([splinea(:,1); splineb(:,1)]));
                miny = floor(min([splinea(:,2); splineb(:,2)]));
                maxy = ceil(max([splinea(:,2); splineb(:,2)]));
                % Now give some frame of 4 pixels (should be enough)
                extrapixels = 4;
                minx = minx-extrapixels;
                maxx = maxx+extrapixels;
                miny = miny-extrapixels;
                maxy = maxy+extrapixels;
                % Check if there are negative values or values exceeding the image size
                if minx<1,                  minx = 1;                     end
                if maxx>myinfotosave.w2,    maxx = myinfotosave.w2;       end
                if miny<1,                  miny = 1;                     end
                if maxy>myinfotosave.w1,    maxy = myinfotosave.w1;       end                
                % Translate the splines to their new position in the small image
                splinebs = [splineb(:,1)-minx+1 splineb(:,2)-miny+1];
                splineas = [splinea(:,1)-minx+1 splinea(:,2)-miny+1];
                % Create a finer grid (divide each pixel in "myf" quadrants)
                myf = 4;
                sidex = myf*(maxx-minx+1);
                sidey = myf*(maxy-miny+1);
                % Scale the splines
                splineas = splineas*myf;
                splinebs = splinebs*myf;
                % Create the new image
                sgrid  = zeros(sidey,sidex);
                % Approximate the splines in the finer grid image
                for npt=1:size(splinebs,1) % Assign a value of 1 to the reference (to create a potential)
                    sgrid(round(splinebs(npt,2)),round(splinebs(npt,1))) = 1; 
                end
                for npt=1:size(splineas,1) % Assign a value of 2 to the no-reference (to create a potential)
                    sgrid(round(splineas(npt,2)),round(splineas(npt,1))) = 2; 
                end
                % Create a mask
                smask = sgrid;
                smask(smask>0) = 1;
                smask = imfill(smask,'holes');
                %-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                % Solve numerically the Laplace's equation 
                PotentialMap = SolveLaplaceEquation2D(sgrid,0); % The second input is a display flag 
                if displayPotentialMap
                    PotMap = PotentialMap;
                    PotMap = PotMap-0.999999999;
                    PotMap(PotMap<0) = 0;
                    figure('NumberTitle','off', ...
                             'Name',['Potential Map - Slice number:  ' num2str(nslice)]);
                    imagesc(PotMap.*smask); 
                    axis equal; 
                    impixelinfo;  colormap(hot);
                    drawnow;
                end
                %-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                % Get the cartilage contour
                cartROI = bwboundaries(smask,8,'noholes');
                cartROI = cartROI{1};
                % Get the first end-point
                pos1 = matchingpointsinC([splinebs(1,2) splinebs(1,1)],cartROI);
                % Reorder the cartilage contour to start at the first point
                if pos1==1 | pos1==size(cartROI,1)
                    cROI = cartROI;  
                else
                    cROI = [cartROI(pos1:end,:); cartROI(1:pos1-1,:)];
                end
                % Find the second end-point 
                pos2 = matchingpointsinC([splinebs(end,2) splinebs(end,1)],cROI);
                % Get the bone border
                splinebs = cROI(1:pos2,:);
                % Get the articular border
                splineas =  cROI(pos2+1:end,:);
                splineas = flipdim(splineas,1);
                % Make them share the end points
                splineas = [splinebs(1,:); splineas; splinebs(end,:)];
                %-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                % Smooth the splines to avoid zig-zags
                splinebs = [smooth(splinebs(:,1),7) smooth(splinebs(:,2),7)];
                splineas = [smooth(splineas(:,1),7) smooth(splineas(:,2),7)];
                %-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                % Get matchings between the bone and the articular surfaces
                Streamlines = getstreamlines2D(PotentialMap,splinebs,2,smask,displayStreamLines); 
                if displayStreamlinesonMap
                    hlay = figure('NumberTitle','off', ...
                                          'Name',['Streamlines on Potential Map - Slice number: ', num2str(nslice)]);
                    smap = nmap(miny:maxy,minx:maxx);
                    smap = imresize(smap,4);
                    smap = smap/max(smap(:));
                    smap(smap<0) = 0;
                    [w1p w2p] = size(smap);
                    npixels = w1p*w2p;
                    PotMap = PotentialMap;
                    PotMap = PotMap-0.999999;
                    PotMap(PotMap<0) = 0;
                    cmap = hot(256);
                    [indmap] = gray2ind(PotMap,256); 
                    ima3 = zeros(w1p,w2p,3);
                    pos = find(smask==1);
                    for n=1:length(pos)
                        ima3(pos(n))=cmap(double(indmap(pos(n))+1),1); 
                        ima3(pos(n)+npixels)=cmap(double(indmap(pos(n))+1),2); 
                        ima3(pos(n)+2*npixels)=cmap(double(indmap(pos(n))+1),3); 
                    end    
                    pos0 = find(smask==0);
                    ima3(pos0) = smap(pos0);
                    ima3(npixels+pos0) = smap(pos0);
                    ima3(2*npixels+pos0) = smap(pos0);
                    figure(hlay);
                    imagesc(ima3);  axis equal; 
                    hold on;  
                    for npt=1:3:length(Streamlines);
                        if isempty(Streamlines{npt})
                            continue;
                        end
                        plot(Streamlines{npt}(:,2),Streamlines{npt}(:,1),'k');
                    end
                    drawnow;
                end
                % Get the number of streamlines
                nstreamlines = length(Streamlines);
                % Get the new splineb and splinea, and put the coordinates back in their original coordinate frame
                splinebs = [];
                splineas = [];
                % Prepare a variable for thickness values
                r = [];
                for npt=1:nstreamlines
                    if isempty(Streamlines{npt})
                        continue; % Go to the next loop
                    end
                    % Get the bone and articular spline
                    splinebs = [splinebs; Streamlines{npt}(1,:)];
                    splineas = [splineas; Streamlines{npt}(end,:)];
                    % Get the thickness based on the streamlines
                    if rstreamlines
                        distba = diff(Streamlines{npt},1,1);
                        distba = sqrt(sum(distba.^2,2));   
                        cumdistba = cumsum(distba);
                        cumdistba = [0; cumdistba];
                        r = [r; cumdistba(end)];
                    end
                end  
                % Get the thicknesses based on Euclidean distances
                if rstreamlines==0  
                    r = sqrt(sum((splineas-splinebs).^2,2));
                end
                % Downsample the splines
                splinebs = splinebs/myf;
                splineas = splineas/myf;
                % Correct the thickness
                r = r/myf;
                % Put them back in the original coordinate frame
                splinebs(:,1) = splinebs(:,1)+miny-1;
                splinebs(:,2) = splinebs(:,2)+minx-1;
                splineas(:,1) = splineas(:,1)+miny-1;
                splineas(:,2) = splineas(:,2)+minx-1;
                % Update splineb and splinea
                splineb = splinebs;
                splinea = splineas;
                % Put again the x-coordinate in the first column and the y-coordinate in the second
                splineb = flipdim(splineb,2);
                splinea = flipdim(splinea,2);              
        end
        % Keep the thicknesses
        Thicks = [Thicks; r];
        %-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        % Get the geodesic length of the bone cartilage interface
        distb = diff(splineb,1,1);
        distb = sqrt(sum(distb.^2,2));   
        cumdistb = cumsum(distb);
        cumdistb = [0; cumdistb];
        lengthb = cumdistb(end);
        %-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        % Get the sampling rate in pixels 
        deltad = 3;
        %-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        % We will flat the cartilage with respect to the first spline point
        fptb = splineb(1,:);            
        %-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        % Get distance segmentes
        cdist = 0:deltad:lengthb;
        diffend = abs(cdist(end)-lengthb);
        if diffend<mean(r)/2
            cdist(end) = lengthb;
        else
            cdist = [cdist lengthb];
        end
        cdist = cdist';     
        %-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        % Get the spline landmarks on the bone-cartilage interface
        matchcd = matchingpointsinC(cdist,cumdistb);        
        cland = splineb(matchcd,:);  
        % Get the number of landmarks in the bone cartilage interface
        nlands = length(cland);
        %-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        % Establish the target landmarks for the bone cartilage interface: 
        if strcmp(mainorientation,'Horizontal') % On a horizontal line at the same "y" that that "fptb" (see few lines above)
            Tland = zeros(nlands,1);
            Tland(:,1) = fptb(1,1);
            Tland(:,2) = fptb(1,2);
            Tland(:,1) = Tland(:,1)+Tlands_direction*cdist; 
        else % On a vertical line at the same "x" that that "fptb" (see few lines above)
            Tland = zeros(nlands,1);
            Tland(:,1) = fptb(1,1);
            Tland(:,2) = fptb(1,2);
            Tland(:,2) = Tland(:,2)+Tlands_direction*cdist;             
        end
        Tlandb = Tland;
        %-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        % This is only for validation of preservation of thickness
        if matching_method==1 || matching_method==2
            wr = cland;
            wr(:,1) = wr(:,1)+r(matchcd).*cos(theta(matchcd))/2;
            wr(:,2) = wr(:,2)+r(matchcd).*sin(theta(matchcd))/2;
        end
        %-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        % Establish the target landmarks for the articular surface trying to preserve the original cartilage thickness
        cland = [cland; splinea(matchcd,:)];          
        Tland =  [Tland; zeros(nlands,2)];       
        if strcmp(mainorientation,'Horizontal') 
            Tland(nlands+1:end,1) = Tland(1:nlands,1);
            if isboneref==1
                Tland(nlands+1:end,2) = Tland(1:nlands,2)+Tlands_direction*r(matchcd);
            else
                Tland(nlands+1:end,2) = Tland(1:nlands,2)-Tlands_direction*r(matchcd);
            end
        else
            Tland(nlands+1:end,2) = Tland(1:nlands,2);
            if isboneref==1
                Tland(nlands+1:end,1) = Tland(1:nlands,1)-Tlands_direction*r(matchcd);
            else
                Tland(nlands+1:end,1) = Tland(1:nlands,1)+Tlands_direction*r(matchcd);
            end
        end
        Tlanda = Tland(nlands+1:end,:);
        %-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        % Remove repeated points to avoid problems with warping
        cland(1,:) = [];
        Tland(1,:) = [];        
        cland(end,:) = [];
        Tland(end,:) = [];         
        %-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        % Establish a sufficiently large support "a" for warping for a smooth deformation
        a = lengthb;
        if displaywsplines
            % This is for validation and visualization purposes only
            [alphasv,av] = getFornefettcoefficients(cland,Tland,a);
            wsplineb = applyForneffetcoeffs(splineb,cland,alphasv,av);
            wsplinea = applyForneffetcoeffs(splinea,cland,alphasv,av);        
            wr = applyForneffetcoeffs(wr,cland,alphasv,av);
        end
        % But do this since we want to do backward mapping
        % Get the warping parameters to be used 
        [alphas,a] = getFornefettcoefficients(Tland,cland,a);  
        %-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        % Get the coordinates to be warped     
        maxx = round(max(Tland(:,1)))+10;     
        maxy = round(max(Tland(:,2)))+10;     
        [tx ty] = ndgrid(1:maxx,1:maxy);        
        tx = tx(:);
        ty = ty(:);
        % Warp the coordinates that include the region of interest
        [trc] = applyForneffetcoeffs([tx ty],Tland,alphas,a);
        %-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        % Get the original coordinates
        [otx oty] = meshgrid(1:myinfotosave.w2,1:myinfotosave.w1);
        % Obtain pixel values of the coordinates that include the region of interest via backward mapping and interpolation
        fmap = interp2(otx,oty,nmap,reshape(trc(:,1),maxx,maxy),reshape(trc(:,2),maxx,maxy),'spline');    
        fmap = fmap';        
        %-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        % Obtain the mask for the flatten cartilage and crop it
        Tsplineb = bezieruniformsampling([Tlandb ones(size(Tlandb,1),1)],myinfotosave.deltax);
        Tsplinea = bezieruniformsampling([Tlanda ones(size(Tlanda,1),1)],myinfotosave.deltax);
        wmask = poly2mask([Tsplineb(:,1); Tsplinea(:,1)],[Tsplineb(:,2); Tsplinea(:,2)],maxy,maxx);
        % Mask the flatten map
        wmap = wmask.*fmap;
        % Avoid Nans
        wmap(isnan(wmap)) = 0;
        % Get the mean of the flattened cartilage
        validpts = wmask>0;
        fT2means = [fT2means; mean(wmap(validpts))];
        fT2stds = [fT2stds; std(wmap(validpts))];
        if 1 % Crop the flattened map
            minr = round(min(Tland(:,2)))-1;   if minr<1,      minr=1;         end
            maxr = round(max(Tland(:,2)))+1; 
            minc = round(min(Tland(:,1)))-1;   if minc<1,      minc=1;         end
            maxc = round(max(Tland(:,1)))+1;
            wmap = wmap(minr:maxr,minc:maxc);
            %-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            % Save new info
            fmyinfotosave.dataperslice{nslice}.xcoordinatesspl{nspline} = Tlandb(:,1)-minc+1;
            fmyinfotosave.dataperslice{nslice}.ycoordinatesspl{nspline} = Tlandb(:,2)-minr+1;  
            fmyinfotosave.dataperslice{nslice}.xcoordinatesspl{nspline+1} = Tlanda(:,1)-minc+1;
            fmyinfotosave.dataperslice{nslice}.ycoordinatesspl{nspline+1} = Tlanda(:,2)-minr+1;   
        else
            % Save new info
            fmyinfotosave.dataperslice{nslice}.xcoordinatesspl{nspline} = Tlandb(:,1);
            fmyinfotosave.dataperslice{nslice}.ycoordinatesspl{nspline} = Tlandb(:,2);  
            fmyinfotosave.dataperslice{nslice}.xcoordinatesspl{nspline+1} = Tlanda(:,1);
            fmyinfotosave.dataperslice{nslice}.ycoordinatesspl{nspline+1} = Tlanda(:,2);  
        end
        %-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        % Adjusts the spline coordinates in case of multiple segments since we are going to stack the cartilage segments (see below)
        if strcmp(mainorientation,'Horizontal')
            fmyinfotosave.dataperslice{nslice}.xcoordinatesspl{nspline} = fmyinfotosave.dataperslice{nslice}.xcoordinatesspl{nspline}+size(sliceMaps,2);    
            fmyinfotosave.dataperslice{nslice}.xcoordinatesspl{nspline+1} = fmyinfotosave.dataperslice{nslice}.xcoordinatesspl{nspline+1}+size(sliceMaps,2);
        else
            fmyinfotosave.dataperslice{nslice}.ycoordinatesspl{nspline} = fmyinfotosave.dataperslice{nslice}.ycoordinatesspl{nspline}+size(sliceMaps,1); 
            fmyinfotosave.dataperslice{nslice}.ycoordinatesspl{nspline+1} = fmyinfotosave.dataperslice{nslice}.ycoordinatesspl{nspline+1}+size(sliceMaps,1);   
        end
        %-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        % Save the cartilage segments 
        if strcmp(mainorientation,'Horizontal')
            sliceMaps = mycat(2,sliceMaps,wmap);
            % Separate disconnected cartilage components
            sliceMaps = [sliceMaps  nan(size(sliceMaps,1),noffsets+1)]; 
        else % Means vertical
            sliceMaps = mycat(1,sliceMaps,wmap);
            % Separate disconnected cartilage components
            sliceMaps = [sliceMaps; nan(noffsets+1,size(sliceMaps,2))];
        end
        %-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        % Display the data
        if displayflag
            if displaywsplines
                % Plot the bone spline
                figure; plot(splineb(:,1),splineb(:,2),'g-'); hold on;
                set(gca,'YDir','Reverse');  axis equal;f
                % Plot the articular spline
                plot(splinea(:,1),splinea(:,2),'r-'); hold on;
                % Plot source landmarks
                plot(cland(:,1),cland(:,2),'g.'); hold on;
                % Plot target landmarks
                plot(Tland(:,1),Tland(:,2),'b.'); hold on;   
                % Plot the warped-bone spline
                plot(wsplineb(:,1),wsplineb(:,2),'b-');
                % Plot the warped-articular spline
                plot(wsplinea(:,1),wsplinea(:,2),'m-');
                % Plot the spline at half thickness
                plot(wr(:,1),wr(:,2),'c.-');
                title(['Slice:' num2str(nslice) '-Spline:' num2str(nspline)]);
            end
            % Display the masked map
            figure; imagesc(cmap);  caxis(gca,[0 100]); axis equal;
            title(['Slice:' num2str(nslice) '-Spline:' num2str(nspline)]);
            % Display the warped map
            figure; imagesc(wmap);  caxis(gca,[0 100]); axis equal;
            title(['Slice:' num2str(nslice) '-Spline:' num2str(nspline)]);
        end % End display
    %-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    end % End spline loop
    % Save the slice cartilage segments
    fIma{nslice} = sliceMaps;
    %-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
end % End of slice loop     
%-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
% Get the new in-plane matrix size
fw1 = 1;
fw2 = 1;
for nslice=1:myinfotosave.w3
    [cw1 cw2] = size(fIma{nslice});
    if cw1>fw1
        fw1 = cw1;
    end
    if cw2>fw2
        fw2 = cw2;
    end
end
% Update fmyinfotosave
fmyinfotosave.w1 = fw1;
fmyinfotosave.w2 = fw2;
%-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
MeanT2OrigFlat = [oT2means fT2means];
StdT2OrigFlat = [oT2stds fT2stds];
% diffMeanT2 = fT2means-oT2means;
%Disp('[Original_Means Flatten_Means Flatten_Means-Original_Means Original_Stds Flatten_Stds]');
%[MeanT2OrigFlat  diffMeanT2 StdT2OrigFlat]
% Disp(num2str([min(diffMeanT2) max(diffMeanT2)]));
% Disp(strcat('Mean of means of ROI gray-level differences between original and flattened cartilage = ',num2str(mean(diffMeanT2))));
% Disp(strcat('Std of means of ROI gray-level differences between original and flattened cartilage = ',num2str(std(diffMeanT2))));
% [hT2s pT2s] = ttest(oT2means,fT2means);
% if hT2s
%     Disp(strcat('Paired t-test indicates that the difference between mean gray-levels of original and flattened cartilage is significant with p=',num2str(pT2s)));
% else
%     Disp(strcat('Paired t-test indicates that the difference between mean gray-levels of original and flattened cartilage is NOT significant with p=',num2str(pT2s)));
% end    
%-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

