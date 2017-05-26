
function newImage = discWaveletTransform2D(image, level, wname)

    % Clear all variables and close all windows
    clear vars;
    close all;
    
    % Apply the discrete 2D wavelet transform
    % Note: uses only the LL result to transform
    for i=1:level
       [img, LH, HL, HH] = dwt2(image, wname);
    end   
    
    % Apply the inverse discrete 2D wavelet transform
    % Used to reconstruct the image
    % Note: uses only the LL result to transform
    for i=1:level
       image = idwt2(image, [], [], [], wname);
    end
    
    newImage = image;
end