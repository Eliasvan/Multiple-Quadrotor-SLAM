% This simulates the scenario of 'p' 3D points seen in 'f' frames.
% For each combination 'p' and 'f', we check there are enough constraints
% to solve for all 3D points (3 unknowns each) and poses (6 unknowns each).
%   => num_unknowns = 3*p + 6*f

num_points3D = 0:150;
num_frames = 0:10;

% Set easy-to-read metrics, fonts, ...
set(0, 'DefaultAxesFontName', 'DejaVuSans')
set(0, 'DefaultTextFontname', 'DejaVuSans')
set(0, 'DefaultFigurePaperUnits', 'inches')
set(0, 'DefaultFigurePaperSize', [4, 4])
set(0, 'DefaultFigurePaperPosition', [0 0 [4, 4]])
set(0, 'DefaultLineMarkerSize', 3)



% First, we simulate only measuring the 2D points of each 3D point in each frame,
% this will add 2 constraints each.
% We don't consider prior knowledge on any of the unknowns, for simplicity.   (= worst case)
%   => num_constraints = 2*p*f

% Plot 2 surfaces: number of unknowns and number of constraints,
% both in function of num_points3D and num_frames.
figure(1)
clf
[XX, YY] = meshgrid(num_points3D, num_frames);
num_unknowns = 3*XX + 6*YY;
num_constraints = 2*XX.*YY;
hold on
mesh(XX, YY, num_unknowns)
mesh(XX, YY, num_constraints)
hold off

% Plot the intersection of both surfaces:
% in the inner region close to the origin, the problem is under-constrained.
figure(2)
clf
p = num_points3D(7:end);
f = num_frames(4:end);
hold on
plot(p, 3*p ./ (2*p - 6), 'r-')
plot(6*f ./ (2*f - 3), f, 'r-')
hold off
xlabel('number of 3D points')
ylabel('number of frames')
axis([0 p(end) 0 f(end)])
%  title('Projective factors : feasibility line')
saveas(gcf, 'figures/projFactors_feasibilityLine.pdf')

figure(1)
hold on
plot3(p, 3*p ./ (2*p - 6),    6*p.*p ./ (2*p - 6), 'r-', 'LineWidth',2)
plot3(6*f ./ (2*f - 3), f,    12*f.*f ./ (2*f - 3), 'r-', 'LineWidth',2)
hold off
view(25, 45)
xlabel('number of 3D points')
ylabel('number of frames')
zlabel('number of unknowns or constraints')
axis([0 p(end) 0 f(end) 0 4000])
%  title('Projective factors : #unknowns vs #constraints')
set(gcf, 'PaperSize', [5, 5], 'PaperPosition', [0 0 [5, 5]]);
saveas(gcf, 'figures/projFactors_unknownsVSconstraints.pdf')

%% Parametrized version of previous figure
%q = 18:(num_points3D(end)*num_frames(end));    % = p.*f
%s = sqrt(1 - 18./q);
%figure(3)
%plot(q./3 .* (1 - s), q./6 .* (1 + s), 'r-');



% Now, we simulate also measuring the odometry between each frame,
% this will add 6 constraints each (assuming the scale is already known).
% We don't consider prior knowledge on any of the unknowns, for simplicity.   (= worst case)
%   => num_constraints = 2*p*f + 6 * (f-1)

% Plot 2 surfaces: number of unknowns and number of constraints,
% both in function of num_points3D and num_frames.
figure(3)
clf
num_constraints = 2*XX.*YY + 6 * (YY-1);
hold on
mesh(XX, YY, num_unknowns)
mesh(XX, YY, num_constraints)
hold off

% Plot the intersection of both surfaces:
% in the inner region close to the origin, the problem is under-constrained.
figure(4)
clf
p = num_points3D(3:end);
f = num_frames(4:end);
hold on
plot(p, (6 + 3*p) ./ (2*p), 'r-')
plot(6 ./ (2*f - 3), f, 'r-')
hold off
xlabel('number of 3D points')
ylabel('number of frames')
axis([0 p(end) 0 f(end)])
%  title('Projective and Odometry factors : feasibility line')
saveas(gcf, 'figures/projANDodometryFactors_feasibilityLine.pdf')

figure(3)
hold on
plot3(p, (6 + 3*p) ./ (2*p),    3*p + 6 * (6 + 3*p) ./ (2*p), 'r-', 'LineWidth',2)
plot3(6 ./ (2*f - 3), f,        18 ./ (2*f - 3) + 6*f, 'r-', 'LineWidth',2)
hold off
view(25, 45)
xlabel('number of 3D points')
ylabel('number of frames')
zlabel('number of unknowns or constraints')
axis([0 p(end) 0 f(end) 0 4000])
%  title('Projective and Odometry factors : #unknowns vs #constraints')
set(gcf, 'PaperSize', [5, 5], 'PaperPosition', [0 0 [5, 5]]);
saveas(gcf, 'figures/projANDodometryFactors_unknownsVSconstraints.pdf')

%% Parametrized version of previous figure
%q = 4:(num_points3D(end)*num_frames(end));    % = p.*f
%figure(5)
%plot((2*q - 6) ./ 3, 3*q ./ (2*q - 6), 'r-');
