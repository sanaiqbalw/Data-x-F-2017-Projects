function newA = mycat(dim,A,B)
% newA = mycat(dim,A,B);
% 
% Function to cat arrays of different size by filling one array with nan's.
% 
% Inputs:
% dim   - Scalar indicating the dimension for the concatenation. 
%         It can only take values of 1 (rows), 2 (columns), or 3 (slices).
% A     - The array to which we will append data.
% B     - The array to be appended to A.
%
% Outputs:
% newA  - A after concatenating B along dimension "dim".
%
% 
%
% by
% Julio Carballido-Gamio
% 2005
% Julio.Carballido@gmail.com
% 

[w1A w2A w3A] = size(A);
[w1B w2B w3B] = size(B);

switch dim
    case 1
        neww1 = w1A+w1B;
        neww2 = max([w2A w2B]);
        neww3 = max([w3A w3B]);
        newA = nan(neww1,neww2,neww3);
        newA(1:w1A,1:w2A,1:w3A) = A;
        newA(w1A+1:end,1:w2B,1:w3B) = B;
    case 2
        neww1 = max([w1A w1B]);
        neww2 = w2A+w2B;
        neww3 = max([w3A w3B]);
        newA = nan(neww1,neww2,neww3);
        newA(1:w1A,1:w2A,1:w3A) = A;
        newA(1:w1B,w2A+1:end,1:w3B) = B;        
    case 3
        neww1 = max([w1A w1B]);
        neww2 = max([w2A w2B]);
        neww3 = w3A+w3B;
        newA = nan(neww1,neww2,neww3);
        newA(1:w1A,1:w2A,1:w3A) = A;
        newA(1:w1B,1:w2B,w3A+1:end) = B;
    otherwise
        error('Not ready for 4 dimensions yet.');
end