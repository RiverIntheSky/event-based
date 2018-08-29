file = '/home/weizhen/Documents/dataset/shapes_rotation/event.txt';
ev=textscan(fopen(file),'%f %f %f %f');
close all
figure;
hold on;
pos = [];
neg = [];
for i = 1:size(ev{1,1},1)
    if ev{1,4}(i) == 1
        pos = [pos;ev{1,1}(i), ev{1,2}(i),ev{1,3}(i)];
    else
        neg = [neg;ev{1,1}(i), ev{1,2}(i),ev{1,3}(i)];
    end
end

plot3(pos(:,1), pos(:,2), 180-pos(:,3),'.','color','r');
plot3(neg(:,1), neg(:,2), 180-neg(:,3),'.','color','b');

im = imread('/home/weizhen/Documents/dataset/shapes_rotation/images/frame_00000000.png');
[X,Y] = meshgrid(1:size(im,2), 1:size(im,1));
x_coord = X(:); y_coord = Y(:);
scatter3(ones(size(x_coord))*0.028046, x_coord, 180-y_coord, 2, repmat(double(im(:)), 1, 3)/255);

im = imread('/home/weizhen/Documents/dataset/shapes_rotation/images/frame_00000016.png');
[X,Y] = meshgrid(1:size(im,2), 1:size(im,1));
x_coord = X(:); y_coord = Y(:);
scatter3(ones(size(x_coord))*0.733092, x_coord, 180-y_coord, 2, repmat(double(im(:)), 1, 3)/255);

im = imread('/home/weizhen/Documents/dataset/shapes_rotation/images/frame_00000022.png');
[X,Y] = meshgrid(1:size(im,2), 1:size(im,1));
x_coord = X(:); y_coord = Y(:);
scatter3(ones(size(x_coord))*0.997484, x_coord, 180-y_coord, 2, repmat(double(im(:)), 1, 3)/255);

im = imread('/home/weizhen/Documents/dataset/shapes_rotation/images/frame_00000026.png');
[X,Y] = meshgrid(1:size(im,2), 1:size(im,1));
x_coord = X(:); y_coord = Y(:);
scatter3(ones(size(x_coord))*1.173745, x_coord, 180-y_coord, 2, repmat(double(im(:)), 1, 3)/255);

set(gca,'ytick',[])
set(gca,'ztick',[])
