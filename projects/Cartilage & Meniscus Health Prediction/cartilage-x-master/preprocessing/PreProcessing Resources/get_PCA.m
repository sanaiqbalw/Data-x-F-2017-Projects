function [eig_vecs eig_vals centroid] = get_PCA(XYZ,displayflag)
% [eig_vecs eig_vals centroid] = get_PCA(XYZ,displayflag);
% 
% Function to perform principal componenet analysis of 2D or 3D data.
%
% Inputs:
% XYZ         -->     Array of doubles with cartesian coordinates of the data to be analyzed. 
%                     2D: [x's   y's]
%                     3D: [x's   y's   z's]
% displayflag -->     displayflag=1 plots eigenvectors on top of data.
%                     displayflag=0 plots nothing.
%
% Outputs:
% eig_vecs    -->     Array of 2(2D) or 3(3D) columns, where each column is an eigenvector in descending order to match "eig_vals".
% eig_vals    -->     Array of 2(2D) or 3(3D) columns, where each column is an eigenvalue in descending order.
% centroid    -->     Array of 2(2D) or 3(3D) elements (row) indicating the coordinates of the centroid.
%
%
%
% by Julio Carballido-Gamio
% 2005
% Julio.Carballido@gmail.com
%


if nargin==1
    displayflag = 0;
end
ndims = size(XYZ,2);
% 1. Bring the center of mass to the origin
npts = size(XYZ,1);
centroid = mean(XYZ,1);
XYZ = XYZ - repmat(centroid,npts,1);
%-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
% 2. Normalize the data
XYZ = XYZ/sum(sqrt(sum(XYZ.^2)));
%-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
% 3. Compute the 3x3 or 2x2 covariance of the data
if ndims==3
    sumx2 = sum(XYZ(:,1).^2);
    sumxy = sum(XYZ(:,1).*XYZ(:,2));
    sumxz = sum(XYZ(:,1).*XYZ(:,3));
    sumy2 = sum(XYZ(:,2).^2);
    sumyz = sum(XYZ(:,2).*XYZ(:,3));
    sumz2 = sum(XYZ(:,3).^2);
    cov_mat = [sumx2  sumxy  sumxz
               sumxy  sumy2  sumyz
               sumxz  sumyz  sumz2];
elseif ndims==2
    sumx2 = sum(XYZ(:,1).^2);
    sumxy = sum(XYZ(:,1).*XYZ(:,2));   
    sumy2 = sum(XYZ(:,2).^2);   
    cov_mat = [sumx2  sumxy  
               sumxy  sumy2];
else
    error('Only nx3 or nx2 vectors please');    
end
%-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
% 4. Compute the eigenvectors, and eigenvalues
[eig_vecs,eig_vals] = eig(cov_mat);
eig_vals = diag(eig_vals);
% Put eigenvalues in descending order (in rows)
eig_vals = flipdim(eig_vals,1);
% Put eigenvalues in columns
eig_vals = eig_vals';
% Rearrange eigenvectors accordingly (in columns)
eig_vecs = flipdim(eig_vecs,2);
%-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
if displayflag
    figure;
    mydelta = 1;
    if ndims==3
        plot3(XYZ(1:mydelta:end,1),XYZ(1:mydelta:end,2),XYZ(1:mydelta:end,3),'k.');
        hold on;
        quiver3(0,0,0,eig_vecs(1,1),eig_vecs(2,1),eig_vecs(3,1),eig_vals(1),'b');
        quiver3(0,0,0,eig_vecs(1,2),eig_vecs(2,2),eig_vecs(3,2),eig_vals(2),'c');
        quiver3(0,0,0,eig_vecs(1,3),eig_vecs(2,3),eig_vecs(3,3),eig_vals(3),'m');
        axis equal;
    elseif ndims==2
        plot(XYZ(1:mydelta:end,1),XYZ(1:mydelta:end,2),'k.');
        hold on;
        quiver(0,0,eig_vecs(1,1),eig_vecs(2,1),eig_vals(1),'b');
        quiver(0,0,eig_vecs(1,2),eig_vecs(2,2),eig_vals(2),'c');
        axis equal;
    end      
end
    
