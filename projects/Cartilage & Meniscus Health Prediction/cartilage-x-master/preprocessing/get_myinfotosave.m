function myinfotosave = get_myinfotosave(image_to_flatten, boundary_in, boundary_out)
load('D:\Google Drive\Berkeley\Courses\Data X IEOR 290\Project\Cartilage-X\PreProcessing Resources\manualExamplesforKosta\ACL037_130222_E318_CUBE_trans_contra_LFC.mat');
myinfotosave.w1 = size(image_to_flatten,1);
myinfotosave.w2 = size(image_to_flatten,2);
myinfotosave.w3 = size(image_to_flatten,3);
myinfotosave.dataperslice = {};
myinfotosave.dataperslice{1}.nsplines = 2;
myinfotosave.dataperslice{1}.xcoordinatesspl{1} = boundary_in(:,1);
myinfotosave.dataperslice{1}.ycoordinatesspl{1} = boundary_in(:,2);
myinfotosave.dataperslice{1}.xcoordinatesspl{2} = boundary_out(:,1);
myinfotosave.dataperslice{1}.ycoordinatesspl{2} = boundary_out(:,2);
myinfotosave.dataperslice{1}.iscartilagesegmented = 1;
myinfotosave.deltax = 1;
myinfotosave.deltay = 1;
myinfotosave.deltaz = 1;
myinfotosave.pathname = pwd;
end
