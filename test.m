numbers = [60000 61000 62000 63000 65000 66000 67000 68000 69000];
    
for j=1:numel(numbers)
    imgDir = sprintf('moreFaces/%i',numbers(j));
    imgList = dir(sprintf('%s/*.png',imgDir));
    nImages = length(imgList);

    for i=1:nImages
        imName = sprintf('%s',imgList(i).name);
        pathToIm = sprintf('%s/%s',imgDir,imName);
        im = rgb2gray(im2single(imread(pathToIm)));
        im = im(1:end-2,1:end-2);
        im = imresize(im, 1/3.5);
        
        saveName = int2str(numbers(j)+i);
        saveTo = sprintf('%s/%s.jpg','moreFaces',saveName);
        imwrite(im,saveTo);
    end
end