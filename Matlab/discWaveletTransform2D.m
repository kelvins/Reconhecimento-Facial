
function newImage = discWaveletTransform2D(img, level, wname)

    % Clear all variables and close all windows
    clear vars
    close all
    
    % Apply the discrete 2D wavelet transform
    % Note: uses only the LL result to transform
    for i=1:level
       [img,LH,HL,HH] = dwt2(img, wname) 
    end   
    
    % Apply the inverse discrete 2D wavelet transform
    % Used to reconstruct the image
    % Note: uses only the LL result to transform
    for i=1:level
       img = idwt2(img,[],[],[], wname)
    end
    
    newImage = img
end