function [alphas,a] = getFornefettcoefficients(Sland,Tland,a)
% [alphas,a] = getFornefettcoefficients(Sland,Tland,a)
%
% Function to calculate Fornefett coefficients.
%
% Inputs:
% Sland   --->  2-column (2D) or 3-column (3D) array with the set of source landmarks where each row represents the coordinates of a landmark.
% Tland   --->  2-column (2D) or 3-column (3D) array with the set of target landmarks where each row represents the coordinates of a landmark.
%               Rows of Sland and Tland correspond to each oter.
% a       --->   Scalar that controls the influence of the landmarks in their corresponding neighborhoods.
% 
%
%
% Julio Carballido-Gamio
% 2004
% Julio.Carballido@gmail.com
%

ndim = size(Sland,2);
r = dist2inC(Sland,Sland);
if isempty(a)
    if ndim==2 % Paper says 2.98 the maximum landmark displacement in the coordinate directions
        a = 2.98*max(max(abs(Sland-Tland))); % 2.98*max(max(abs(Sland-Tland)))
    else % Paper says 3.66 the maximum landmark displacement in the coordinate directions
        a = 3.66*max(max(abs(Sland-Tland))); %10*2.98 for cartilage interpolation; 3.66 for elastic registration of bones at 1.5 T.; 1.5 for T2 and T1rho
    end
end

%a = a*a1f;

r = r/a;
pos = find(r<0 | r>=1);
K = ((1-r).^4).*(4*r+1);
K(pos) = 0;
alphas = inv(K)*(Tland-Sland);




