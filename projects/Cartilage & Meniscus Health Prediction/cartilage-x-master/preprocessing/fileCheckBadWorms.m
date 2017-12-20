cd /data/bigbone4/DeepLearning_temp/Data/CartilageX/mri/flat_LFC/
dirLFC= dir;
worms = [];
listLFC = {};
for k=3:size(dirLFC,1)
   filename = dirLFC(k).name;
   fin=fopen(filename,'rt');
   I=fread(fin, 100*512*512,'single');
   worms(length(worms)+1) = I(1);
   a = strsplit(filename,'.raw');
   listLFC{length(listLFC)+1} = a{1};
end 
  
load('/data/bigbone4/ciriondo/clean_all_path.mat')
fileIDlist={};
for k =2:size(raw,1)
    split_file = strsplit(raw{k,1},{'/','.mat'}); fileID = split_file{end-1};
    fileIDlist{k-1} = fileID;
end

lesion_index_LFC = [];
for j = 1:length(fileIDlist)
    fileID = fileIDlist{j};
    if ~ismember(fileID,listLFC)
        % check if worms >0
        if raw{j+1,3} >0
            lesion_index_LFC(length(lesion_index_LFC)+1) = j+1;
        end
    end
    
end
cd /data/bigbone4/ciriondo/
save('lesion_index_LFC.mat', 'lesion_index_LFC');
