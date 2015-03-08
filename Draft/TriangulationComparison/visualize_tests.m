%%
% Test 1
%%

load test_1and2
triangl_methods = cellstr(triangl_methods);
num_traj = length(trajectories);
num_triangl_methods = length(triangl_methods);
num_points = length(points_3D);

l = cell(0);    % legend
t = cell(0);;    % title
for i = 1 : num_traj
    for j = 1 : num_triangl_methods
        l{j, i} = [strrep(strrep(triangl_methods{j}, '_', ' '), 'triangulation', ''), ': ', 'Trajectory ', num2str(i)];
    end
    t{i} = ['Trajectory ', num2str(i), ': ', trajectories{i}.traj_descr];
end

% Plot RMS vs X- and Y- coordinate

mesh_coord_max = max(max(points_3D(:, 1:3)));
mesh_coords = (-mesh_coord_max : mesh_coord_max)';
mesh_values = NaN * ones(2*mesh_coord_max + 1);    % create mesh with initially invalid values
idxs = int64(points_3D(:, 1:3)) + mesh_coord_max + 1;

for j = 1 : num_triangl_methods
    figure(1)
    clf
    colormap copper

    hold on
    for i = 1 : num_traj
        for k = 1:size(idxs, 1)
            mesh_values(idxs(k, 1), idxs(k, 2)) = p_err3D_median_summary(i, j, k);
        end
        mesh(mesh_coords, mesh_coords, mesh_values)
    end
    hold off

    view(45, 45)
    set(gca, 'zscale', 'log')
    set(gca, 'ZMinorTick', 'on')
    xlabel('X-coordinate of 3D point')
    ylabel('Z-coordinate of 3D point')
    zlabel('Root Median Squared 3D error')
    legend(l(j, :))

    saveas(gcf, ['figures/test1_', triangl_methods{j}, '_3Dplot.pdf'])
end

% Plot error ellipsoids for (almost) each point of Iterative LS triangulation

mean = zeros(3, 1);
covar = zeros(3, 3);

j = find(strcmp(triangl_methods, 'iterative_LS_triangulation'));
if length(j)    % iterative_LS exists
    for i = 1 : num_traj
        
        figure(2)
        clf
        colormap summer

        hold on
        for p = 1 : num_points
            mean(:) = p_err3Dv_mean_summary(i, j, p, :);
            covar(:, :) = p_err3Dv_covar_summary(i, j, p, :, :);
            if (any(isnan(mean)) || any(isinf(mean)) || any(isnan(covar(:))) || any(isinf(covar(:))))
                continue
            end
            
            % plot line from real point to mean of various noise trials of the triangulated point
            p_mid = points_3D(p, 1:3)' + mean;
            plot3([points_3D(p, 1); p_mid(1)], [points_3D(p, 2); p_mid(2)], [points_3D(p, 3); p_mid(3)], 'r-')
            
            [vectors, values] = eig(covar);
            values = diag(values);
            vectors = vectors * diag(sign(values));
            values = sqrt(abs(values));

            [x, y, z] = ellipsoid(0,0,0, values(1),values(2),values(3), 30);

            for k = 1:31
                res = vectors * [x(k, :); y(k, :); z(k, :)];
                x(k, :) = res(1, :);
                y(k, :) = res(2, :);
                z(k, :) = res(3, :);
            end

            x = p_mid(1) + x;
            y = p_mid(2) + y;
            z = p_mid(3) + z;

            % plot covariance of various noise trials of the triangulated point
            h = surfl(x, y, z);
            set(h, 'edgecolor','none')
        end
        hold off

        axis(mesh_coord_max * [-1, 1, -1, 1, -1, 1])
        view(40, 40)
        xlabel('X-coordinate of 3D point')
        ylabel('Y-coordinate of 3D point')
        zlabel('Z-coordinate of 3D point')
        title(['Error ellipsoids of triangulated points for end-pose of trajectory ', num2str(i)])

        saveas(gcf, ['figures/test1_vectors_traj', num2str(i), '_3Dplot.pdf'])
    end
end



%%
% Test 2
%%

triangl_line_styles = {'-', '--', ':', '-.'};
traj_color_styles = {'b', 'g', 'r', 'c'};

% 3D Plot by visualizing the trajectories

figure(3)
clf

traj_x = zeros(num_poses, num_traj);
traj_z = zeros(num_poses, num_traj);
for i = 1 : num_traj
    traj_x(:, i) = trajectories{i}.sideways_values;
    traj_z(:, i) = trajectories{i}.towards_values - default_parameters.cam_pose_offset;
    if strcmp(trajectories{i}.traj_descr, 'From 1st cam, describing circle (while facing the sphere of points) until intersecting with trajectory 3')
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

view(45, 45)
set(gca, 'zscale', 'log')
xlabel('X-coordinate of 2nd camera center')
ylabel('Z-coordinate of 2nd camera center')
zlabel('Root Median Squared 3D error')
legend(['Projection of 4th trajectory on the ground'; l(:)])
title(strjoin(t, ' \n'))

saveas(gcf, 'figures/test2_err3D_3Dplot.pdf')

% 2D Plot, in function of trajectory progress

figure(4)
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

saveas(gcf, 'figures/test2_err3D_2Dplot.pdf')

% Robustness Plot of Iterative LS triangulation

j = find(strcmp(triangl_methods, 'iterative_LS_triangulation'));
if length(j)    % iterative_LS exists

    % False positives

    figure(5)
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

    saveas(gcf, 'figures/test2_robustnessFalsePositivesRatio_plot.pdf')

    % False negatives

    figure(6)
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

    saveas(gcf, 'figures/test2_robustnessFalseNegativesRatio_plot.pdf')
    
end



%%
% Test 3
%%

load test_3
triangl_methods = cellstr(triangl_methods);
noise_type_descr = cellstr(noise_type_descr);
num_noise_types = length(noise_type_descr);
noise_sigma_values = repmat(noise_sigma_values, [1, num_noise_types]);

j = find(strcmp(triangl_methods, 'iterative_LS_triangulation'));
if length(j)    % iterative_LS exists
    
    % Plot of RMS 3D error in function of noise sigma

    for i = 1 : num_traj
        figure(7)
        clf

        plot(noise_sigma_values, squeeze(err3D_median_summary(i, :, :, j))')

        xlabel('camera 2D noise sigma value [px]')
        ylabel('Root Median Squared 3D error')
        legend(noise_type_descr(:))
        title([strrep(triangl_methods{j}, '_', ' '), ' on end-pose of trajectory ', num2str(i)])

        saveas(gcf, ['figures/test3_err3D_traj', num2str(i), '.pdf'])
    end
end
