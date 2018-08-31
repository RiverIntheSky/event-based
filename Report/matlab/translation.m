
close all
clear all
% boxes_rotation num_rot
file = '/home/weizhen/Documents/dataset/boxes_translation/planar/50000/';

groundtruth = fopen(strcat(file, 'groundtruth_translation.txt'));
cg=textscan(groundtruth,'%f %f %f %f');
estimated = fopen(strcat(file, 'estimated_translation.txt'));
ce=textscan(estimated,'%f %f %f %f');


lost = 1060;
scale = 1;   
% scale = 0.76;

angles = zeros(lost, 2);

error = zeros(lost, 3);
error_angle = zeros(lost, 1);
maxx = 0;

abss = zeros(lost, 1);
traveled_distance = 0;
last = [0,0,0];
rms = 0;
for i = 1:lost
    g = [cg{1,2}(i), cg{1,3}(i), cg{1,4}(i)];
    if (abs(g(1)) > 10 || abs(g(2)) > 10 || abs(g(3)) > 10)
        cg{1,2}(i) = cg{1,2}(i-1);
        cg{1,3}(i) = cg{1,3}(i-1);
        cg{1,4}(i) = cg{1,4}(i-1);
        g = [cg{1,2}(i), cg{1,3}(i), cg{1,4}(i)];
    end
    traveled_distance = traveled_distance + norm(g-last);
    norm(g-last);
    last = g;
    e = [ce{1,2}(i), ce{1,3}(i), ce{1,4}(i)];
    ce{1,2}(i) = ce{1,2}(i)*scale;
    ce{1,3}(i) = ce{1,3}(i) *scale;
    ce{1,4}(i) = ce{1,4}(i)*scale;

    error(i, 1) = abs( ce{1,2}(i) -cg{1,2}(i));
    error(i, 2) = abs( ce{1,3}(i) -cg{1,3}(i));
    error(i, 3) = abs( ce{1,4}(i) -cg{1,4}(i));
    abss = abss+norm(error(i,:))^2;
    error_angle(i, 1) = abs(norm(g) - norm(e));
    angle = atan2d(norm(cross(g,e)),dot(g,e));
    if (norm(g) > maxx)
        maxx = norm(g);
    end
    
    angles(i, 1) = ce{1,1}(i);
    
  
    if (g(1)>10 || g(2)>10 ||g(3)>10 )
        angle = 26;  
    end
        angles(i, 2) = angle;
end
traveled_distance
err = mean(error)

% figure(1)
% hold on
% histogram(angles(:, 2),linspace(1, 180, 100));
% % title('angle between estimation and ground truth')
% % legend('translation','rotation')
% xlabel('degree')
% 
% figure(2)
% hold on;
% h3 = plot(cg {1,1},cg{1,2},'color',[0.85,0.33,0.1],'linewidth',2);
% h6 = plot(ce{1,1},ce{1,2},'color',[0.0,0.45,0.74]);ylabel('rad/s')
% axis([43,47.5,-0.6,0.4])
% 

figure(3)

subplot(3, 1, 1)
hold on;
h1 = plot(cg{1,1}(1:lost),cg{1,2}(1:lost),'color',[0.85,0.33,0.1],'linewidth',2);
h4 = plot(ce{1,1}(1:lost),ce{1,2}(1:lost),'color',[0.0,0.45,0.74]);ylabel('m')
% axis([0, 59, -1, 1])
% yyaxis right
% h9= plot(ce{1,1},error(:,1),'color',[0.1,0.1,0.1])
legend([h1,h4], 'ground truth', 'estimation')
% axis([0, 59, 0, 2])
title('x')
xlabel('t')


subplot(3, 1, 2)
hold on;
h2 = plot(cg{1,1}(1:lost),cg{1,3}(1:lost),'color',[0.85,0.33,0.1],'linewidth',2);
h5 = plot(ce{1,1}(1:lost),ce{1,3}(1:lost),'color',[0.0,0.45,0.74]);ylabel('m')
% axis([0, 59, -1, 1])
% yyaxis right
% plot(ce{1,1},error(:,2),'color',[0.1,0.1,0.1])
% axis([0, 59, 0, 2])
% legend([h2,h5], 'ground truth', 'estimation')

title('y')
xlabel('t')


subplot(3, 1, 3)
hold on;
h3 = plot(cg {1,1}(1:lost),cg{1,4}(1:lost),'color',[0.85,0.33,0.1],'linewidth',2);
h6 = plot(ce{1,1}(1:lost),ce{1,4}(1:lost),'color',[0.0,0.45,0.74]);ylabel('m')
% axis([0, 59, -1, 1])
% yyaxis right
% plot(ce{1,1},error(:,3),'color',[0.1,0.1,0.1])
% axis([0,59, 0, 2])

% legend([h3,h6], 'ground truth', 'estimation')
% axis([0, ce{1,1}(90), -1, 1])
title('z')
xlabel('t')
% figure(4)
% hold on;
% 
% a = 263;
% b = 797;
% h1=plot(cg{1,1}(a:b)-0.01,cg{1,2}(a:b),'r','linewidth',2);
% plot(ce{1,1}(a:b),ce{1,2}(a:b),'-.r');ylabel('deg')
% h2=plot(cg{1,1}(a:b)-0.01,cg{1,3}(a:b),'g','linewidth',2);
% plot(ce{1,1}(a:b),ce{1,3}(a:b),'-.g');ylabel('deg')
% h3=plot(cg{1,1}(a:b)-0.01,cg{1,4}(a:b),'b','linewidth',2);
% plot(ce{1,1}(a:b),ce{1,4}(a:b),'-.b');ylabel('deg')
% legend([h1,h2,h3],'pitch', 'yaw','roll')


