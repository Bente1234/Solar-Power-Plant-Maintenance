
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import gurobipy as grb
from scipy.stats import gamma

#Maintenance of solar power plant
#Task a
# Read Data
df_west = pd.read_csv('data/spu_efficiency_West.csv')
df_east = pd.read_csv('data/spu_efficiency_East.csv')

#Restrict to days <= 600 because every 200 days they get cleaned and the data only goes up to 730 so after 600 there is no full degradation cycle
df_west = df_west[df_west["Day"] <= 600].reset_index(drop=True)

#Replacing efficiency values that are <= 0 or > 1 with Nan, these are invalid so should be replaced later
eff = df_west.copy()
eff.iloc[:, 1:] = eff.iloc[:, 1:].applymap(
    lambda x: np.nan if (pd.isna(x) or x <= 0 or x > 1) else x
)

#Linear interpolation of all missing values
eff.iloc[:, 1:] = eff.iloc[:, 1:].interpolate(method="linear")

#For each cycle (200 days), the day after maintenance the efficiency should be 1, when handling Nan values with linear interpolation it could be that a value is filled in for the day after maintenance that is not 1, this is resolved in this manner
reset_days = [1, 201, 401]  
reset_days = [d for d in reset_days if d in eff["Day"].values]
eff.loc[eff["Day"].isin(reset_days), eff.columns[1:]] = 1.0

#Now the degradation is modelled by: 1 - efficiency
deg_west = 1 - eff.iloc[:, 1:].values

#Empty list for all degradation increments
all_increments = []

#Loop through all maintenance cycles (0–200, 200–400, 400–600) that are in scope
for start in [0, 200, 400]:
    #Select the data in the cyle
    cycle = eff[(eff["Day"] > start) & (eff["Day"] <= start + 200)]
    deg_cycle = 1 - cycle.iloc[:, 1:].values

    #Computing daily degradation increments
    delta_x = np.diff(deg_cycle, axis=0)
    #Flattening increments into a 1D array such that all SPUs can be considered
    inc_cycle = delta_x.flatten()
    
    #Keep only positive increments
    inc_cycle = inc_cycle[inc_cycle > 0]
    
    all_increments.extend(inc_cycle)

#Conversion to an array
inc_west = np.array(all_increments)

#Parameter estimation with MoM based on the corrected degradation data
#Sample mean
mu_hat = np.mean(inc_west)
#Sample variance
sigma2_hat = np.var(inc_west, ddof=1)

alpha_hat = mu_hat**2 / sigma2_hat
beta_hat = mu_hat / sigma2_hat
alp = alpha_hat
bet = beta_hat

print(f"Task (a) MoM, alpha = {alp:.2f}, beta = {bet:.4f}")

# Task d
#RUL estimation with Montecarlo Simulation
RUL = []

#Preprocessing the data in the same way as in question a (removing values below zero or above one and replacing them by Nan)
eff_east = df_east.copy()
try:
    eff_east.iloc[:, 1:] = eff_east.iloc[:, 1:].map(
        lambda x: np.nan if (pd.isna(x) or x <= 0 or x > 1) else x
    )
except AttributeError:
    eff_east.iloc[:, 1:] = eff_east.iloc[:, 1:].applymap(
        lambda x: np.nan if (pd.isna(x) or x <= 0 or x > 1) else x
    )

#Linear interpolation of the missing values and the values > 1 or <= 0
eff_east.iloc[:, 1:] = eff_east.iloc[:, 1:].interpolate(method="linear")


#Converting the efficiencies to degradation values and taking only the positive
deg_east = 1 - eff_east.iloc[:, 1:].values
#Computing the daily increments of degradation
delta_x_east = np.diff(deg_east, axis=0)
#Flattening increments into a 1D array such that all SPUs can be considered
inc_east = delta_x_east.flatten()
inc_east = inc_east[inc_east >= 0]  

#Method of Moments to estimate the Gamma parameters of the East based on the corrected input data
#Sample mean
mu_east = np.mean(inc_east)
#Sample variance
sigma2_east = np.var(inc_east, ddof=1)
alpha_east = mu_east**2 / sigma2_east
beta_east = mu_east / sigma2_east

print(f"Task (d) MoM, alpha = {alpha_east:.2f}, beta = {beta_east:.4f}")

current_eff = eff_east.iloc[-1, 1:].values

#After computing the parameters, a Montecarlo simulation is used to estimate the RUL of the East
threshold = 0.8 #Threshold below which the SPU should be cleaned
n_sim = 1000 #Number of simulation rounds

RUL = []

#Looping over each SPU 
for eff0 in current_eff:
    degradation = 1 - eff0
    rul_samples = []

    #Performing the Monte Carlo Simulation
    for _ in range(n_sim):
        t = 0
        deg = degradation #Current degradation state
        while deg < 0.2:  #Simulate until the degradation >= 0.2 (that is equal to efficiency <= 0.8)
            inc = gamma.rvs(a=alpha_east, scale=1/beta_east) #Drawing a random increment from the Gamma parameters estimated above
            deg += inc #Update degradation
            t += 1  #Increase the day by one
        rul_samples.append(t) #Storing the simulated RUL

    #Taking the average over all RUL's
    RUL.append(np.mean(rul_samples))

print(f"Task (d) Average RUL of East area: {np.mean(RUL):.2f} days")

# Task e
#Read data
df_cost = pd.read_csv('data/cost_cleaning.csv')
df_rul = pd.read_csv('data/RUL_North.csv')

#Creating parameters for the optimisation
C_P = df_cost['C_P, power price'].to_numpy()
C_D = df_cost['C_D, daily charge'].to_numpy()
C_U = df_cost['C_U, unit charge'].to_numpy()
RUL_i = df_rul['RUL (days)'].to_numpy()

max_clean = 3  #maximum number of cleaning jobs per day
I = RUL_i.size  #number of SPU's from the data set
T = df_cost.shape[0]  #number of considered days

#Creating a Gurobi model
m = grb.Model("CleaningSchedule")

#Decision variables
x = m.addVars(I, T, vtype=grb.GRB.BINARY, name="x")  #A binary variable that is 1 if a SPU is cleaned on day t and 0 if not
y = m.addVars(T, vtype=grb.GRB.BINARY, name="y")     #A binary variable that is 1 if any SPU is cleaned on day t and 0 if none

#Objective function which is to minimize the total costs
m.setObjective(
    grb.quicksum(C_D[t] * y[t] for t in range(T)) +
    grb.quicksum((C_U[t] + C_P[t]) * x[i, t]
                 for i in range(I) for t in range(T)),
    grb.GRB.MINIMIZE
)

#Constraints
#Each SPU should be cleaned before it's RUL
for i in range(I):
    deadline = min(int(RUL_i[i]), T)  
    m.addConstr(
        grb.quicksum(x[i, t] for t in range(deadline)) == 1,
        name=f"RUL_{i}"
    )

#Ensures that the maximum number of cleaning jobs in one day is smaller or equal to 3
for t in range(T):
    m.addConstr(grb.quicksum(x[i, t] for i in range(I)) <= max_clean, name=f"Cap_{t}")

#Ensures that y is at most 1
for t in range(T):
    m.addConstr(grb.quicksum(x[i, t] for i in range(I)) <= max_clean * y[t], name=f"Link_{t}")

#Solving the problem with the solver
m.setParam('OutputFlag', 0)
m.optimize()

print("Optimal cleaning schedule (SPU, Day):")
#Creating an alternative representation of the schedule to ensure clarity
rows = []
for t in range(T):
    cleaned_today = [i+1 for i in range(I) if x[i, t].X > 0.5]
    day_label = t + 1
    if cleaned_today:
        rows.append({
            "Day": day_label,
            "Jobs": len(cleaned_today),
            "SPUs": ", ".join(map(str, cleaned_today))
        })

# Convert to DataFrame
schedule_df = pd.DataFrame(rows)

# Print nicely
print(schedule_df.to_string(index=False))

#Print optimal objective value following from the optimisation problem
print(f"(f) Optimal objective value: {m.objVal:.2f}")
