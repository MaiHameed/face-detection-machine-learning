%% Using rectangle to draw bounding boxes
im = imread('class.jpg');
figure;
hold on;
imshow(im);
rectangle('Position',[800 100 300 600],'EdgeColor','g');

%%
im = single(imread('class.jpg'));
cellSize = 6;
hog = vl_hog(im, cellSize, 'verbose');
imhog = vl_hog('render', hog, 'verbose');
figure;
imshow(imhog);
colormap gray;

%%
figure;
A = [1,1;
    1,10;
    10,10;
    10,1];
plot(A(:,1),A(:,2),'-g');