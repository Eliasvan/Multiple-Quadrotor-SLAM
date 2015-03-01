%% 
% Test 1
%%


%% 
% Test 2
%%

load test2
traj_description = cellstr(traj_description);
triangl_methods = cellstr(triangl_methods);
num_poses = parameters.num_poses;
num_traj = length(traj_description);
num_triangl_methods = length(triangl_methods);

% 3D Plot by visualizing the trajectories

triangl_line_styles = {'-', '--', ':', '-.'};
traj_color_styles = {'b', 'g', 'r', 'c'};

l = cell(0);    % legend
t = cell(0);;    % title
for i = 1 : num_traj
    for j = 1 : num_triangl_methods
        l{j, i} = [strrep(strrep(triangl_methods{j}, '_', ' '), 'triangulation', ''), ': ', 'Trajectory ', num2str(i)];
    end
    t{i} = ['Trajectory ', num2str(i), ': ', traj_description{i}];
end

figure(1)
clf

max_sideways = parameters.max_sideways;
max_towards = parameters.max_towards;
max_angle = parameters.max_angle;
cam_pose_offset = default_parameters.cam_pose_offset;

sideways_values = linspace(0, max_sideways, num_poses);
towards_values = linspace(0, max_towards, num_poses);
angle_values = linspace(0, max_angle, num_poses);

traj_x = zeros(num_poses, num_traj);
traj_z = zeros(num_poses, num_traj);
for i = 1 : num_traj
    if     strcmp(traj_description{i}, 'From 1st cam, to sideways')
        traj_x(:, i) = sideways_values;
        traj_z(:, i) = zeros(size(towards_values)) - cam_pose_offset;
    elseif strcmp(traj_description{i}, 'From 1st cam, towards the sphere of points')
        traj_x(:, i) = zeros(size(sideways_values));
        traj_z(:, i) = towards_values - cam_pose_offset;
    elseif strcmp(traj_description{i}, 'From last pose of trajectory 1, towards the sphere of points, parallel to trajectory 2')
        traj_x(:, i) = sideways_values(end) * ones(size(sideways_values));
        traj_z(:, i) = towards_values - cam_pose_offset;
    elseif strcmp(traj_description{i}, 'From 1st cam, describing circle (while facing the sphere of points) until intersecting with trajectory 3')
        traj_x(:, i) = cam_pose_offset * sin(angle_values);
        traj_z(:, i) = cam_pose_offset * -cos(angle_values);
        plot3(traj_x(:, i), traj_z(:, i), min(err3D_median_summary(:)) * ones(size(traj_z(:, i))), 'k-')    % projection on the ground
    end
end

hold on
for i = 1 : num_traj
    for j = 1 : num_triangl_methods
        plot3(traj_x(:, i), traj_z(:, i), err3D_median_summary(i, :, j), [traj_color_styles{i}, triangl_line_styles{j}])
    end
end
hold off

set(gca, 'zscale', 'log')
xlabel('X-coordinate of 2nd camera center')
ylabel('Z-coordinate of 2nd camera center')
zlabel('Root Median Squared 3D error')
legend(['Projection of 4th trajectory on the ground'; l(:)])
title(strjoin(t, ' \n'))

saveas(gcf, 'test2_err3D_3Dplot.pdf')

% 2D Plot, in function of trajectory progress

figure(2)
clf

progress = linspace(0, 100, num_poses);

hold on
for i = 1 : num_traj
    for j = 1 : num_triangl_methods
        semilogy(progress, err3D_median_summary(i, :, j), [traj_color_styles{i}, triangl_line_styles{j}])
    end
end
hold off

set(gca, 'YMinorTick', 'on')
xlabel('progress of trajectory [%]')
ylabel('Root Median Squared 3D error')
legend(l(:))

saveas(gcf, 'test2_err3D_2Dplot.pdf')

% Robustness Plot of Iterative LS triangulation

j = find(strcmp(triangl_methods, 'iterative_LS_triangulation'));
if length(j)    % iterative_LS exists

    % False positives

    figure(3)
    clf
    
    hold on
    for i = 1 : num_traj
        plot(progress, false_pos_summary(i, :, j), [traj_color_styles{i}, '-']);
    end
    hold off

    set(gca, 'YMinorTick', 'on')
    xlabel('progress of trajectory [%]')
    ylabel('Ratio of amount of false positives')
    legend(l(j, :))
    title(['Robustness (3D error distance threshold = ', num2str(robustness_thresh_max), '), "positive" means properly triangulated'])

    saveas(gcf, 'test2_robustnessFalsePositivesRatio_plot.pdf')

    % False negatives

    figure(4)
    clf
    
    hold on
    for i = 1 : num_traj
        plot(progress, false_neg_summary(i, :, j), [traj_color_styles{i}, '-']);
    end
    hold off

    set(gca, 'YMinorTick', 'on')
    xlabel('progress of trajectory [%]')
    ylabel('Ratio of amount of false negatives')
    legend(l(j, :))
    title(['Robustness (3D error distance threshold = ', num2str(robustness_thresh_min), '), "positive" means properly triangulated'])

    saveas(gcf, 'test2_robustnessFalseNegativesRatio_plot.pdf')
    
end


%% 
% Test 3
%%
