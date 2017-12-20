function targetcoords = applyForneffetcoeffs(Scoords,Slandmarks,alphas,a)
% targetcoords = applyForneffetcoeffs(Scoords,Slandmarks,alphas,a)
% 
% Function to apply the Forneffet coefficients to a set of coordinates.
%
% Inputs:
% Scoords      - 2-column (2D) or 3-column (3D) array with the coordinates to be warped.
%                First colum are x's, second column are y's, and third column are z's (if 3D).
% Slandmarks   - 2-column (2D) or 3-column (3D) array with the set of source landmarks where each row represents the coordinates of a landmark.
% alphas       - Array of same size as Slandmarks with the corresponding coefficients for warping.
% a            - Scalar that controls the influence of the landmarks in their corresponding neighborhoods.
%
% Outputs:
% targetcoords - The transformed Scoords.
%
%
%
% by
% Julio Carballido-Gamio
% 2005
% Julio.Carballido@gmail.com
%

r = dist2inC(Scoords,Slandmarks);
r = r/a;
pos = find(r<0 | r>=1);
phi = ((1-r).^4).*(4*r+1);
phi(pos) = 0;
targetcoords = Scoords+phi*alphas;