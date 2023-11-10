# EconModel-MorrisAnalysis
## Problem
In this assignment, the focus is on applying the Morris method, a computational economics model with numerous parameters. The Morris method aims to determine which parameters significantly influence the model's outcomes without the need for exhaustive calibration. The evaluation is based on a goodness function, specifically the Euclidean distance of average values from the target values in the second half of the simulation run. The targets include the average employment rate, average wage share, and average banking share, each calculated with respect to specified benchmarks.

## Solution
The provided script implements the Morris method for sensitivity analysis in a simulation model, focusing on the impact of various parameters on model outcomes. To use the code, users can set baseline parameter values in the `params` dictionary, specify the number of trajectories, run the simulation, and then employ the Morris method for analysis. The method returns two crucial results: goodness differences, representing the impact of each parameter on the goodness function, and a parameter importance ranking. Larger positive/negative goodness differences indicate stronger positive/negative impacts, while the ranking reflects the relative importance of each parameter based on average simulation results. In the provided results, the average goodness difference is negative, suggesting a lower average goodness in simulated trajectories compared to the initial one. The maximum goodness difference being close to zero indicates model stability. The parameter importance ranking highlights "banking_share_initial" as the most influential parameter, followed by "employment_rate_initial" and "wage_share_initial." These insights guide further investigation and calibration efforts for significant model parameters.
