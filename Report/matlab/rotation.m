
close all
clear all
% boxes_rotation num_rot
file = '/home/weizhen/Documents/dataset/dynamic_6dof/translation_direction_planar/50000/';
groundtruth = fopen(strcat(file, 'groundtruth_rotation.txt'));
cg=textscan(groundtruth,'%f %f %f %f');
estimated = fopen(strcat(file, 'estimated_rotation.txt'));
ce=textscan(estimated,'%f %f %f %f');

a = 1;
b = size(cg{1,1},1);
for i = 1:4
cg{1,i} = cg{1,i}(a:b);
ce{1,i} = ce{1,i}(a:b);
end
lost = size(cg{1,1},1);
scale = 180.0/pi;   
% scale = 0.9;
% scale

angles = zeros(lost, 2);

error = zeros(lost, 3);
error_angle = zeros(lost, 1);
maxx = 0;

abss = zeros(lost, 1);
traveled_distance = 0;
last = [0,0,0];
rms = 0;
ang = [];
for i = 1:lost
            cg{1,2}(i) = cg{1,2}(i)*scale;
    cg{1,3}(i) = cg{1,3}(i) *scale;
    cg{1,4}(i) = cg{1,4}(i)*scale;
    g = [cg{1,2}(i), cg{1,3}(i), cg{1,4}(i)];
    if (abs(g(1)) > 1000 || abs(g(2)) > 1000 || abs(g(3)) > 1000)
        cg{1,2}(i) = cg{1,2}(i-1);
        cg{1,3}(i) = cg{1,3}(i-1);
        cg{1,4}(i) = cg{1,4}(i-1);
        g = [cg{1,2}(i), cg{1,3}(i), cg{1,4}(i)];
    end
    e = [ce{1,2}(i), ce{1,3}(i), ce{1,4}(i)];
%     scale = norm(g) / norm(e);
    ce{1,2}(i) = ce{1,2}(i)*scale;
    ce{1,3}(i) = ce{1,3}(i) *scale;
    ce{1,4}(i) = ce{1,4}(i)*scale;

    error(i, 1) = abs( ce{1,2}(i) -cg{1,2}(i));
    error(i, 2) = abs( ce{1,3}(i) -cg{1,3}(i));
    error(i, 3) = abs( ce{1,4}(i) -cg{1,4}(i));
    rms = rms+norm(error(i,:));
    error_angle(i, 1) = abs(norm(g) - norm(e));

    ang = [ang,atan2d(norm(cross(g,e)),dot(g,e))];

    if (norm(g) > maxx)
        maxx = norm(g);
    end
    
    angles(i, 1) = ce{1,1}(i);
%     angles(i, 2) = angle;
end
% 
err = median(error)
histogram(ang,linspace(1, 200, 100));
med = median(ang)
% rms = rms/lost
% figure(1)
% hold on
% plot(scale,med,'*')
% end
% figure(1)
% hold on
% histogram(angles(:, 2),linspace(1, 180, 100));
% % title('angle between estimation and ground truth')
% % legend('translation','rotation')
% xlabel('degree')
% 
% figure(2)
% hold on;
% h3 = plot(cg {1,1}(1:lost),cg{1,4}(1:lost)*scale,'color',[0.85,0.33,0.1],'linewidth',2);
% h6 = plot(ce{1,1}(1:lost),ce{1,4}(1:lost),'color',[0.0,0.45,0.74]);ylabel('rad/s')
% legend('ground truth', 'estimation')
% axis([21,25,-0.2,0.2])
% 
figure(3)

subplot(3, 1, 1)
hold on;
h1 = plot(cg{1,1}(1:lost),cg{1,2}(1:lost),'color',[0.85,0.33,0.1],'linewidth',2);
h4 = plot(ce{1,1}(1:lost),ce{1,2}(1:lost),'color',[0.0,0.45,0.74]);ylabel('deg')
% axis([0, 59, -1, 1])
% yyaxis right
% h9= plot(ce{1,1},error(:,1),'color',[0.1,0.1,0.1])
legend([h1,h4], 'ground truth', 'estimation')
% axis([0, 59, 0, 2])
title('pitch')
xlabel('t')


subplot(3, 1, 2)
hold on;
h2 = plot(cg{1,1}(1:lost),cg{1,3}(1:lost),'color',[0.85,0.33,0.1],'linewidth',2);
h5 = plot(ce{1,1}(1:lost),ce{1,3}(1:lost),'color',[0.0,0.45,0.74]);ylabel('deg')
% axis([0, 59, -1, 1])
% yyaxis right
% plot(ce{1,1},error(:,2),'color',[0.1,0.1,0.1])
% axis([0, 59, 0, 2])
% legend([h2,h5], 'ground truth', 'estimation')

title('yaw')
xlabel('t')


subplot(3, 1, 3)
hold on;
h3 = plot(cg {1,1}(1:lost),cg{1,4}(1:lost),'color',[0.85,0.33,0.1],'linewidth',2);
h6 = plot(ce{1,1}(1:lost),ce{1,4}(1:lost),'color',[0.0,0.45,0.74]);ylabel('deg')
% axis([0, 59, -1, 1])
% yyaxis right
% plot(ce{1,1},error(:,3),'color',[0.1,0.1,0.1])
% axis([0,59, 0, 2])

% legend([h3,h6], 'ground truth', 'estimation')
% axis([0, ce{1,1}(90), -1, 1])
title('roll')
xlabel('t')

% figure('DefaultAxesFontSize',20)
% hold on;
% 
% a = 1480;
% b = 1730;
% h1=plot(cg{1,1}(a:b)-0.01,cg{1,2}(a:b),'r','linewidth',2);
% plot(ce{1,1}(a:b),ce{1,2}(a:b),'-.r');ylabel('m')
% h2=plot(cg{1,1}(a:b)-0.01,cg{1,3}(a:b),'g','linewidth',2);
% plot(ce{1,1}(a:b),ce{1,3}(a:b),'-.g');
% h3=plot(cg{1,1}(a:b)-0.01,cg{1,4}(a:b),'b','linewidth',2);
% plot(ce{1,1}(a:b),ce{1,4}(a:b),'-.b');
% legend([h1,h2,h3],'x', 'y','z')




