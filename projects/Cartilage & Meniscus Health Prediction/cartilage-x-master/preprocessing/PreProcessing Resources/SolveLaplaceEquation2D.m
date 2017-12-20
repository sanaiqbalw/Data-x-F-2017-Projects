function PotentialMap = SolveLaplaceEquation2D(Mask,displayflag)
% PotentialMap = SolveLaplaceEquation2D(Mask,displayflag)
% 
% Inspired from:
% Tobin Fricke, 2005-10-05 <tobin at pas dot rochester dot edu>
%
% This function numerically solves the 2D Laplace's Equation.
% The equation is also known as the Potential Equation:
% uxx + uyy = 0
% where:
% u = u(x,y)
% ut = 0
% So basically we have Dirilecht boundary conditions.
% We have fixed the stopping condition to 2500 iterations because the stopping condition based on difference of potentials sometimes is not working.
% 
% Inputs:
% Mask          - It is a 2D array of doubles. Mask has a border section with values equal to n, and a border section with values equal to m (n~=m).
% displayflag   - displayflag=1 Displays the current result every 10 iterations .
%                 displayflag=0 Displays nothing.
%
% Outputs       
% Potentialmap  - It is a 2D array of doubles with the same size as Mask with the numerical solution to the Laplace's Equation.
%
%
% 
% by
% Julio Carballido-Gamio
% 2010
% Julio.Carballido@gmail.com
%

% Check for existence of inputs
if ~exist('Mask','var'),            PotentialMap = []; return;      end
if ~exist('displayflag','var'),     displayflag = 0;                end

% Create a figure if displayflag
if displayflag==1
    figure;
    maxval = max(Mask(:));
end

%Establish the maximum number of iterations
max_iteration = 2500;

% Initialize the potential map
PotentialMap = Mask;
% Find non-zero voxels which must establish the initial conditions
pos = find(Mask);
    
% Create an average kernel that excludies the point of interest
kernel8 = [1/8 1/8 1/8; 1/8 0 1/8; 1/8 1/8 1/8];

% Create a counter
iteration = 0;
while iteration<max_iteration
    % Increment the counter
    iteration = iteration + 1;
    
    % Enforce initial boundary conditions
    PotentialMap(pos) = Mask(pos);
    
    % Draw what we have
    if (mod(iteration,10)==0) && (displayflag==1)
      imagesc(PotentialMap); axis equal; colorbar; drawnow; caxis([0 maxval]);
    end
    
    % Incrementally solve Laplace's equation by setting every cell equal to the average value of the neighboring cells    
    
    %TempMap = 0;
    %for x=[-1 0 1]
    %   for y=[-1 0 1]
    %       if (x~=0 || y~=0)
    %           TempMap = TempMap + circshift(PotentialMap,[x y]);
    %       end
    %   end
    %end
    %PotentialMap = TempMap/8;
    
    % Faster
    PotentialMap = conv2(PotentialMap,kernel8,'same');
end
% Enforce the initial boundary conditions
PotentialMap(pos) = Mask(pos);

