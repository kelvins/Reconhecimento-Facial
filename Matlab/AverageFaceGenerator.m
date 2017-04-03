
% Clear all variables and close all windows
clearvars;
close all;

% Default image size
width  = 100;
height = 100;

% File path
filePath  = 'C:\Users\x\Desktop\base1\';

% Dir and file format
srcFiles  = dir( strcat(filePath, '*.png') );

% For each file in the file path
for i = 1 : length(srcFiles)
    
    file = strcat(filePath, srcFiles(i).name);
    img = imread( file );
    img = imresize(img, [width height]);
    
    if i == 1
        sumImage = double(img); 
    else
        sumImage = sumImage + double(img);
    end
end

% Calculate the average image
sumImage = sumImage / length(srcFiles);
image = mat2gray(uint8(sumImage));
imwrite(image, strcat(filePath, 'AverageFace.png'));
