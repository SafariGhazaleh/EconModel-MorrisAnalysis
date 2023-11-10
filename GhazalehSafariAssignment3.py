#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ghazaleh Safari 

Integrated International Master- and PhD program in Mathematics 
"""

import matplotlib.pyplot as plt
import numpy as np

""" Simulation class. Contains the model and handles its simulation."""
class Simulation_Model():
    def __init__(self,
                 productivity_growth = 0.17,
                 population_growth = 0.0,
                 depreciation_rate = 0.4,
                 capital_output_ratio = 0.8,
                 interest_rate = 0.04,
                 debt_function_parameter = 1.1,
                 philipps_curve_exponent = 1,
                 philipps_curve_factor = 0.2,
                 wage_share_initial = 0.65,
                 employment_rate_initial = 0.9,
                 banking_share_initial = 0.5,
                 t_max = 100,
                 dt = 0.01):
        """
        Constructor method.

        Parameters
        ----------
        productivity_growth : float, optional
            Productivity growth parameter (alpha in Keen 1995). The default is 0.17.
        population_growth : float, optional
            Population growth parameter (beta in Keen 1995). The default is 0.0.
        depreciation_rate : float, optional
            Depreciation rate parameter (gamma in Keen 1995). The default is 0.4.
        capital_output_ratio : numeric, optional
            Capital-output-ratio parameter (nu in Keen 1995). The default is 0.8.
        interest_rate : float, optional
            Interest rate parameter (r in Keen 1995). The default is 0.04.
        debt_function_parameter : numeric, optional
            Debt function parameter (k in Keen 1995). The default is 1.1.
        philipps_curve_exponent : numeric, optional
            Employment rate exponent in the Philipps curve function. The 
            default is 1.
        philipps_curve_factor : numeric, optional
            Employment rate factor in the Philipps curve function. The 
            default is 0.2.
        wage_share_initial : float, optional
            Initial value of the wage share state variable (omega in Keen 1995). 
            The default is 0.65.
        employment_rate_initial : float, optional
            Initial value of the employment rate state variable (lambda in Keen
            1995). The default is 0.9.
        banking_share_initial : float, optional
            Initial value of the banking share of the economy state variable
            (d in Keen 1995). The default is 0.5.
        t_max : numeric, optional
            Length of the simulation. The default is 100.
        dt : numeric, optional
            Step size (granularity of the simulation). The default is 0.01.

        Returns
        -------
        None.

        """
                
        """ Record parameters"""
        self.productivity_growth = productivity_growth 
        self.population_growth = population_growth
        self.depreciation_rate = depreciation_rate
        self.capital_output_ratio = capital_output_ratio
        self.interest_rate = interest_rate
        self.debt_function_parameter = debt_function_parameter
        self.wage_share_initial= wage_share_initial
        self.employment_rate_initial = employment_rate_initial
        self.banking_share_initial = banking_share_initial
        self.alpha = productivity_growth
        self.beta = population_growth  
        self.gamma = depreciation_rate
        self.nu = capital_output_ratio
        self.r = interest_rate
        self.kappa = debt_function_parameter
        self.philipps_curve_exponent = philipps_curve_exponent
        self.philipps_curve_factor = philipps_curve_factor
        self.t_max = t_max
        self.dt = dt
        
        """ Initialize state variables"""
        """ Wage share of income"""        
        self.w = wage_share_initial
        """ Employment rate"""
        self.v = employment_rate_initial    
        """ Banking share of the economy"""
        self.d = banking_share_initial
        
        """ Prepare history records"""
        self.history_t = []
        self.history_v = []
        self.history_w = []
        self.history_d = []

    def philipps_curve(self):
        """
        Philipps curve method.

        Returns
        -------
        numeric
            Value of the Philipps curve term used in the development equation
            of the wage share state variable.

        """
        return self.v**self.philipps_curve_exponent \
                        * self.philipps_curve_factor

    def f_y(self):
        """
        Debt function method. The function is used in the development equations
        of both the employment rate state variable and the banking share of the 
        economy state variable.

        Returns
        -------
        res : numeric
            Function value.

        """
        res = self.kappa / self.nu**2 * (1- self.w**1.1 - self.r*self.d**1.1)
        return res


    def run(self):
        """
        Run method. Handles the time iteration of the simulation

        Returns
        -------
        None.

        """

        for t in range(int(self.t_max / self.dt)):
            """ Wage share change"""
            dw = self.philipps_curve() * self.w - self.alpha * self.w
        	
            """ Employment rate change"""
            dv = self.f_y() * self.v * (1 - self.v**100) + \
                            (- self.alpha - self.beta - self.gamma) * self.v
            
            """ Banking share change"""
            dd = self.d * (self.r - self.f_y() + self.gamma) + \
                                        self.nu * self.f_y() - (1 - self.w)
            
            """ Compute absolutes"""
            self.w += dw * self.dt
            self.v += dv * self.dt
            self.d += dd * self.dt
            
            """ Make sure state variables are not out of bounds"""
            self.ensure_state_validity()
            
            """ Record into history"""
            self.history_w.append(self.w)
            self.history_v.append(self.v)
            self.history_d.append(self.d)
            self.history_t.append(t / self.dt) 

    def ensure_state_validity(self):
        """
        Method for ensuring state validity. Checks that all state variables are
        still in their valid areas (between 0 and 1). It corrects the state
        variables otherwise to allow the simulation to continue gracefully.

        Returns
        -------
        None.

        """
        if self.w < 0:
            self.w = 0.001
        elif self.w > 1:
            self.w = 0.999
        if self.v < 0:
            self.v = 0.001
        elif self.v > 1:
            self.v = 0.999
        if self.d < 0:
            self.d = 0.001
        elif self.d > 1:
            self.d = 0.999

    def return_results(self, show_plot=True):
        """
        Method for returning and visualizing results

        Returns
        -------
        simulation_history : dict
            Recorded data on the simulation run.

        """
        
        """ Prepare return dict"""                
        simulation_history = {"history_t": self.history_t,
                              "history_w": self.history_w,
                              "history_v": self.history_v,
                              "history_d": self.history_d}
        
        """ Create figure showing the development of the simulation in six
            subplots."""
        if show_plot:            
            fig, ax = plt.subplots(nrows=2, ncols=3, squeeze=False)
            ax[0][0].plot(self.history_t, self.history_v)
            ax[0][0].set_xlabel("Time")
            ax[0][0].set_ylabel("Employment rate")
            ax[0][1].plot(self.history_t, self.history_w)
            ax[0][1].set_xlabel("Time")
            ax[0][1].set_ylabel("Wage share")
            ax[0][2].plot(self.history_t, self.history_d)
            ax[0][2].set_xlabel("Time")
            ax[0][2].set_ylabel("Banking share of the ec.")
            ax[1][0].plot(self.history_d, self.history_v)
            ax[1][0].set_xlabel("Banking share of the economy")
            ax[1][0].set_ylabel("Employment rate")
            ax[1][1].plot(self.history_d, self.history_w)
            ax[1][1].set_xlabel("Banking share of the ec.")
            ax[1][1].set_ylabel("Wage share")
            ax[1][2].plot(self.history_w, self.history_v)
            ax[1][2].set_xlabel("Wage share")
            ax[1][2].set_ylabel("Employment rate")
            plt.tight_layout()
            plt.savefig("business_cycle_simulation.pdf")
            plt.show()

        return simulation_history

class MorrisMethod:
    """Default parameter values"""
    params = {
        "productivity_growth": 0.1,
        "population_growth": 0.02,
        "depreciation_rate": 0.4,
        "capital_output_ratio": 1.5,
        "interest_rate": 0.05,
        "debt_function_parameter": 2.0,
        "philipps_curve_exponent": 2.5,
        "philipps_curve_factor": 0.8,
        "wage_share_initial": 0.5,
        "employment_rate_initial": 0.8,
        "banking_share_initial": 0.3,
    }

    def __init__(self, model, params, num_trajectories):
        self.model = model
        self.params = params
        self.num_trajectories = num_trajectories
        
        """# Parameter value ranges for sampling"""
        self.parameter_arrangement = {
            "productivity_growth": [0.05, 0.25, "float"],
            "population_growth": [0.0, 0.05, "float"],
            "depreciation_rate": [0.2, 0.6, "float"],
            "capital_output_ratio": [0.2, 2.5, "float"],
            "interest_rate": [0.0, 0.1, "float"],
            "debt_function_parameter": [0.5, 3.0, "float"],
            "philipps_curve_exponent": [0.9, 3.0, "float"],
            "philipps_curve_factor": [0.01, 1.5, "float"],
            "wage_share_initial": [0.25, 0.75, "float"],
            "employment_rate_initial": [0.6, 0.95, "float"],
            "banking_share_initial": [0.25, 0.9, "float"],
        }
        self.parameter_names = list(self.parameter_arrangement.keys())
        
        """Example current EU values"""
        self.current_eu_values = [0.1, 0.03, 0.35, 1.2, 0.04, 1.8, 2.0, 0.75, 0.85, 0.4]  # Example current EU values

    def generate_samples(self):
        """
        Generate parameter samples for each trajectory.
        
        Returns:
        samples (list): List of parameter samples for each trajectory.
        """
        samples = []
        for _ in range(self.num_trajectories):
            sample = []
            for param, values in self.parameter_arrangement.items():
                """Generate a random parameter value within the specified range"""
                value = np.random.uniform(values[0], values[1]) if values[2] == 'float' else np.random.choice(values)
                sample.append(value)
            samples.append(sample)
        return samples

    def goodness_euclidian(self, simulation_history):
        """
        Function for computing the goodness from Simulation_Model results.
        The goodness here is the Euclidean distance of average values of all
        three dependent variables from target values in the second half of
        the simulation. The first half may contain a transient (before the
        trajectory settles into a stable situation), so the result may be
        better without that part.

        Parameters
        ----------
        simulation_history : dict
            Recorded data on the simulation run.

        Returns
        -------
        goodness : float
            Euclidean distance of the average simulated values from target
            values

        """

        """ Target average employment rate (for EU 2023: unemployment rate
            of 7%, thus 1 - 0.07 = 0.93)"""
        tv = 0.93
        """ Target average wage share (for EU 2021: 197m employees times
            average wage 33500 Euros divided by total GDP of 14.5 trillion
            Euros (data from OECD):
            197 * 33500 / (14.5 * 1000000) = 0.455)"""
        tw = 0.455
        """ Target average banking share (for EU 2021 GDP of 14.5 trillion,
            of which are financial and insurance activities (ISIC rev.4
            section K) 600bn Euros and real estate activities (ISIC rev.4
            section L) 1.4 trillion Euros (data from OECD), thus:
            (600 + 1400) / 14500 = 0.138)"""
        td = 0.138

        """Half the simulation runtime"""
        n = len(simulation_history["history_t"]) // 2

        """ Averages in the second half of the simulation (removing the
            potentially transient first half"""
        avg_employment_rate = np.mean(simulation_history["history_v"][-n:])
        avg_wage_share = np.mean(simulation_history["history_w"][-n:])
        avg_banking_share = np.mean(simulation_history["history_d"][-n:])

        """ Euclidean distance from target"""
        goodness = -((tv - avg_employment_rate)**2 +
                     (tw - avg_wage_share)**2 +
                     (td - avg_banking_share)**2)**0.5
        return goodness

    def analyze_samples(self):
        """Generate parameter samples"""
        samples = self.generate_samples()
        """Store simulation results"""
        results = []
        """Store goodness differences"""
        goodness_values = []
        for i in range(self.num_trajectories):
            sample = samples[i]
            goodness_diffs = []
            for j, (param, _) in enumerate(self.params.items()):
                """Store original parameter value"""
                original_value = self.model.__dict__[param] 
                """Set the parameter value to the sampled value"""
                self.model.__dict__[param] = sample[j]
                """Run the simulation with the updated parameter value"""
                self.model.run()
                """Get the simulation results"""
                result = self.model.return_results(show_plot=False)
                """Store the results"""
                results.append(result)
                """Calculate the goodness of the simulation"""
                goodness = self.goodness_euclidian(result)
                """Calculate goodness difference"""
                goodness_diff = goodness - goodness_values[-1] if goodness_values else 0  
                goodness_diffs.append(goodness_diff)
                """Restore original parameter value"""
                self.model.__dict__[param] = original_value  
            """ Store the maximum goodness difference for each trajectory"""
            goodness_values.append(max(goodness_diffs))  
        return results, goodness_values



if __name__ == '__main__':
    params = {
        "productivity_growth": 0.1,
        "population_growth": 0.02,
        "depreciation_rate": 0.4,
        "capital_output_ratio": 1.5,
        "interest_rate": 0.05,
        "debt_function_parameter": 2.0,
        "philipps_curve_exponent": 2.5,
        "philipps_curve_factor": 0.8,
        "wage_share_initial": 0.5,
        "employment_rate_initial": 0.8,
        "banking_share_initial": 0.3,
    }
    num_trajectories = 10

    S = Simulation_Model()
    S.run()
    result = S.return_results()

    M = MorrisMethod(S, params, num_trajectories)
    results, goodness_values = M.analyze_samples()

    print("Impact of Morris method on goodness:")
    print("Average goodness difference:", np.mean(goodness_values))
    print("Maximum goodness difference:", max(goodness_values))

    importance_ranking = sorted(zip(M.parameter_names, np.mean(list(results[0].values()), axis=0)), key=lambda x: x[1], reverse=True)
    print("\nParameter importance ranking:")
    for param, importance in importance_ranking:
        print(f"{param}: {importance}")
