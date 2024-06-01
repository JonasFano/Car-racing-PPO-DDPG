clc; clear;
results_original_reward = load('results_original_reward.txt');

results_optimized_reward = load('results_optimized_reward.txt');


mean_original = mean(results_original_reward);
mean_optimized = mean(results_optimized_reward);

std_original = std(results_original_reward);
std_optimized = std(results_optimized_reward);

% Display descriptive statistics
fprintf('Original Dataset:\nMean: %.2f, Standard Deviation: %.2f\n', mean_original, std_original);
fprintf('Optimized Dataset:\nMean: %.2f, Standard Deviation: %.2f\n', mean_optimized, std_optimized);


% Perform the Mann-Whitney U test
[p, h, stats] = ranksum(results_original_reward, results_optimized_reward);

% Display the results
if h
    fprintf('There is a significant difference between the two sets of data (p = %.6f)\n', p);
else
    fprintf('There is no significant difference between the two sets of data (p = %.6f)\n', p);
end
