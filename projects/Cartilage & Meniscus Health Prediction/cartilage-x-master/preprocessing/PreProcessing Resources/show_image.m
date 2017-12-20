function show_image(img, show_type)
switch show_type
    case 'imshow'
        imshow(img);
    case 'image'
        image(64*img);
        colorbar;
    case 'imagesc'
        imagesc(img);
    otherwise
        error('show_type should be imshow or image!!');
end
end