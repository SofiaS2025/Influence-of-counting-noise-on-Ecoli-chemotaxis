import numpy as np
from numba import njit,jit
import sys

s=int(sys.argv[1]) #an argument that can be put to seed the random number generator
np.random.seed(s)
# if i want to pass more arguments then 
# s1 = float(sys.argv[1]) this would be the first argument
# s2 = float(sys.argv[2]) this would be the second argument
# etc

dt_model = 5 * 10**(-3) #s   

T_exp = 700 #s
time_points = int(T_exp/ dt_model) + 1 #the number of time points is different from the number of steps. In the array (0,1,2) 
    #there are 2 steps and 3 points
    #if we wanna include in the count both the edges, because our starting point number is 0 (if it was 1 that is not necessary)
    
# expressing randomness outside of the numba enviroment to handle properly the random seed and random number generator
randomness_metnoise = np.random.normal(0,1, size = time_points)

sigma_r = 1
randomness_arrival_rate = np.random.normal(0,sigma_r, size = time_points)


phi0_random = np.random.uniform()
theta0_random = np.random.uniform()

randomness_diff_DR = np.zeros((time_points, 3))
randomness_diff_DT = np.zeros((time_points, 3))

z_random = np.zeros(time_points)


for i in range(0, time_points):

    randomness_diff_DR[i] = np.random.normal(0,1,3)

    randomness_diff_DT[i] = np.random.normal(0,1,3)

    z_random[i] = np.random.uniform()


for i in range(0, time_points):

    randomness_diff_DR[i] = np.random.normal(0,1,3)

    randomness_diff_DT[i] = np.random.normal(0,1,3)

    sigma_r = 1
    randomness_arrival_rate[i] = np.random.normal(0,sigma_r)

    z_random[i] = np.random.uniform()



@njit
def crossp_3d(a, b):
    c = [a[1]*b[2] - a[2]*b[1],
         a[2]*b[0] - a[0]*b[2],
         a[0]*b[1] - a[1]*b[0]]

    return c

@njit
def norm_3d(a):
    norm_squared = a[0]**2 + a[1]**2 + a[2]**2
    return np.sqrt(norm_squared)


@njit
def direction_update_with_D_R(p_t, D_R, dt, randomness_DR):

    # p_t vector

    Gamma_p = np.zeros(3) #vector
    p_updated = np.zeros(3)
    coeff = np.sqrt(2 * D_R )

    Gamma_p = np.sqrt(dt) * randomness_DR

    cross_product = crossp_3d(p_t, Gamma_p)

    delta_p = coeff * np.array(cross_product)
    
    p_updated = p_t + delta_p # vector # p_t and delta_p are perpendicular 

    p_updated_norm = norm_3d(p_updated)

    p_updated_normalized = p_updated / p_updated_norm #vector
    
    return p_updated_normalized 


@njit
def direction_update_with_D_T(p_t, D_T, dt, randomness_DT):
    # p_t vector

    Gamma_p = np.zeros(3) #vector
    p_updated = np.zeros(3)
    coeff = np.sqrt(2 * D_T)

    Gamma_p = np.sqrt(dt) * randomness_DT
    
    # this for loop should be equivalent to taking a random sample from a 3D normal distribution 
    # mean = 0
    # cov = np.identity(3) 
    # size = 1 #how many points do u want to get - i need one point (x,y,z) to associate with each componet of Gamma_p
    # Gamma_p = np.sqrt(dt) * np.random.multivariate_normal(mean, cov, size) #np.random.multivariate_normal() is not supported in numba

    cross_product = crossp_3d(p_t, Gamma_p)

    delta_p = coeff * np.array(cross_product)

    p_updated = p_t + delta_p # vector # p_t and delta_p are perpendicular 

    p_updated_norm = norm_3d(p_updated)

    p_updated_normalized = p_updated / p_updated_norm #vector

    return p_updated_normalized


@njit
def conc_gradient_exp(c_0, g, x):
    return  c_0 * np.exp(g*x)

#def signal_update(c_0, c, t, dt): # signal vector has size = size(c) - 1
    #delts_c = c[t] - c[t-1]
    #return (1/c_0)* (delts_c / dt)
#       OR

@njit
def signal_update(v_0, g, dir): 
    return g * v_0 * dir

@njit
def arrival_rate(kappa_D, r_0, dt_model, c, randomness_arr_rate):

    xinoise_value = np.sqrt(r_0)*np.sqrt(dt_model)*randomness_arr_rate

    return kappa_D*(c) + xinoise_value

@njit
def methylation_noise(time_points, dt_model, D_n, tau_n,randomness_metnoise):

    eta_wnoise_value = np.sqrt(2*D_n)*np.sqrt(dt_model)*randomness_metnoise
    met_noise_eta = np.zeros(time_points)
    d_eta = np.zeros(time_points)

    for t in range(0, time_points-1): 

        d_eta[t] = -(1/tau_n)*dt_model*met_noise_eta[t] + eta_wnoise_value[t]

        met_noise_eta[t + 1] = met_noise_eta[t] + d_eta[t]

    return met_noise_eta


@njit
def probability_run_tumble(t, dt, met_noise_eta, rate_r, alpha_Y, w_rate, epsilon_2, epsilon_3, tau_1, tau_2, K_rate, G_r, r_0, a_0):


    t_prior_values = np.arange(0, t + 1)  # All t_prime values from 0 up to t

    # Compute kernel responses for all t_prime values for this t
    exp_term_1 = np.exp(-(1/tau_1) * (t - t_prior_values) * dt)  
    exp_term_2 = np.exp(-(1/tau_2) * (t - t_prior_values) * dt)

    kernel_lin_responses = G_r * ((1/tau_1) * exp_term_1 - (1/tau_2) * exp_term_2)

    rate_diffs = rate_r - r_0 #rate differences

    increments = kernel_lin_responses * rate_diffs * dt #vector size of t_prime_values
    int_lin_response = np.sum(increments) # sum up all increments for this t 

    activity_a = a_0 - int_lin_response # + met_noise_eta[t]

    #   concentration of the response regulator CheY-P
    Y_conc_CheYP = alpha_Y*activity_a

    #   motor (?)
    G_Y = (epsilon_2/4) - (epsilon_3/2) * (Y_conc_CheYP / (Y_conc_CheYP + K_rate))

    #  switching rates
    lambda_R = w_rate*np.exp(-G_Y)
    lambda_T = w_rate*np.exp(+G_Y)
    

    #probability to be running (steady state case)
    #prob_r = lambda_T / (lambda_T + lambda_R) 

    #probability to continue running after t + dt (Poisson process)
    prob_survival = 1 - lambda_R * dt

    #probability to switch from tumble to run
    prob_t_to_r = lambda_T * dt

    return Y_conc_CheYP, lambda_R, lambda_T, activity_a, prob_survival, prob_t_to_r 

# define a function that eliminates the duplicates values of x that are reported when the bacterium tumbles (which means it is at x still)



@njit
def chemotaxis(dt_model,time_points,phi0_random,theta0_random,randomness_diff_DR,randomness_diff_DT,randomness_arrival_rate,z_random,randomness_metnoise):


    #CONCENTRATION - SIGNAL
    #conc_c = np.zeros((N_time_points, 1))
    c_0 = 1  #muM
    kappa_D = 1.2*(10**5) #s-1 muM-1

    #ARRIVAL RATE
    #rate_r = np.zeros((N_time_points, 1))
    #rate_r[0] = 
    r_0 = kappa_D*c_0 #s-1
    
    g = 0 * 10**(-3) # micrometers-1 (1 micro m = 10^-3 mm)
    v_0 = 20 # micrometers / s : maximal run speed is 20 microm / s

    #ACTIVITY
    #activity_a = np.zeros((N_time_points, 1))
    a_0 = 1/3
    tau_1 = 0.35 #s
    tau_2 = 7.4 #s
    G_r = (1/kappa_D)*2.28

    #METHYLATION NOISE
    D_n = 8.1 *(10**(-4))#+/- 0.9 x 10-4 ??? #s-1
    tau_n = 8.7 #s


    #PARAMETERS

    alpha_Y = 6 #microM
    w_rate = 1.3 #s-1
    epsilon_2 = 40
    epsilon_3 = 40
    K_rate = 3.06 #microM

    #ROTATIONAL DIFFUSION
    D_R = 0.062 # s-1  std = 0.03 s-1
    D_T = 37 * D_R 

    
    met_noise_eta = methylation_noise(time_points, dt_model, D_n, tau_n, randomness_metnoise)

    logic_run_tumble = [] # logic vector: for each time step 1-running 0-tumbling

    dirx_run = []
    dirx_run.append(0)

    position_X = [] #np.zeros(N_time_points) #X position - changes after every run
    position_X.append(0)  # Initialize the first position, starting position (0,0)

    signal_s = []
    signal_s.append(0) # how to initialize the signal

    conc_c = []
    conc_c.append(c_0)

    rate_r = []
    rate_r.append(r_0)

    activity_a = []
    activity_a.append(a_0) 

    x = 0
    dirx_run = [] #directiona along x - changes with every tumble - is the same for consecutive runs

    t_length_each_run = [] # time-lenght of each run
    t_length_each_tumble = [] # time-lenght of each tumble

    num_runs = 0
    index_time = 0
    t_model = []
    t_model.append(index_time) # so t_model has units of index! 

    theta_0 =2*np.pi * theta0_random
    phi_0 = 2*np.pi * phi0_random

    dirx_run_updated = np.cos(theta_0) * np.cos(phi_0)
    dir_3d = [dirx_run_updated, np.cos(theta_0) * np.sin(phi_0), np.sin(theta_0)]
    dir_3d = np.array(dir_3d)

    run_after_tumble_idx = 0 # the first redirection happens with D_R

    j1 = 0
    j2 = 1

    length_single_run = 0 # length in time units of each single run
    length_single_tumble = 0
        

    for index_time in range(1, time_points):

        t_model.append(index_time) #update the time (which keeps running btw)

        if j1 < 1:

            num_runs += 1

            length_single_run += dt_model # it starts with a run from 0 to dt

            logic_run_tumble.append(1)

            run_after_tumble_idx += 1

            if run_after_tumble_idx > 1: 

                dir_3d = direction_update_with_D_R(dir_3d, D_R, dt_model, randomness_diff_DR[index_time])
                dirx_run_updated = dir_3d[0] # x component
                dirx_run.append(dirx_run_updated)


            x += dirx_run_updated*v_0*dt_model
            position_X.append(x)

            conc_c_updated = conc_gradient_exp(c_0, g, x) #updated concentration
            conc_c.append(conc_c_updated)

            signal_s_updated = signal_update(v_0, g, dirx_run_updated)
            signal_s.append(signal_s_updated)

            rate_r_updated = arrival_rate(kappa_D, r_0, dt_model, conc_c_updated, randomness_arrival_rate[index_time])
            rate_r.append(rate_r_updated)
            
            # make sure to pass arrays to a fuction otherwise it will create errors with numba
            Y_conc_CheYP, lambda_R, _, activity_a_updated, prob_survival, prob_t_to_r = probability_run_tumble(index_time, 
                            dt_model, met_noise_eta, np.array(rate_r), alpha_Y, w_rate, epsilon_2, epsilon_3, tau_1, tau_2, K_rate, G_r, r_0, a_0)

            
            activity_a.append(activity_a_updated)

            
            

            if prob_survival < z_random[index_time]: #if the prob to run is smaller than the random number then it tumbles
                t_length_each_run.append(length_single_run)
                
                j1 = 1
                j2 = 0 
                length_single_run = 0

                continue # skip the rest of the loop and starts a new iteration


        if j2 < 1:
            
            length_single_tumble += dt_model

        
            # signal and conc stays the same (same position)
            position_X.append(x)
            conc_c.append(conc_c_updated)

            logic_run_tumble.append(0)

            #redirection after each tumbling ? 
            dir_3d = direction_update_with_D_T(dir_3d, D_T, dt_model, randomness_diff_DT[index_time])
            dirx_run_updated = dir_3d[0] # x component 

            dirx_run.append(0)
            signal_s.append(0) # signal during tumbling goes to 0

            #update again the rate and activity and probability becasue of noise they can still change
            rate_r_updated = arrival_rate(kappa_D, r_0, dt_model, conc_c_updated, randomness_arrival_rate[index_time])
            rate_r.append(rate_r_updated)
            
            Y_conc_CheYP, _, lambda_T, activity_a_updated, _ , prob_t_to_r = probability_run_tumble(index_time, 
                            dt_model, met_noise_eta, np.array(rate_r), alpha_Y, w_rate, epsilon_2, epsilon_3, tau_1, tau_2, K_rate, G_r, r_0, a_0)

            activity_a.append(activity_a_updated)
            

            if prob_t_to_r > z_random[index_time]: 
                t_length_each_tumble.append(length_single_tumble)
                length_single_tumble = 0

                #dirx_run_updated = dirx_run_updated + functions.direction_x_update_with_D_T(D_T, dt_model) # running direction from a tumbling state
                dirx_run.append(dirx_run_updated)
                run_after_tumble_idx = 0
                
                j2 = 1 
                j1 = 0

                continue
    
    # at the end of the loop the bacteria might still be running or tumbling, so there is still need to append those values
    if j1 < 1 and length_single_run > 0:
        t_length_each_run.append(length_single_run)

    if j2 < 1 and length_single_tumble > 0:
        t_length_each_tumble.append(length_single_tumble)

    # Ensure t_model is a NumPy array
    t_model = np.array(t_model)
    #t_model = t_model * dt_model #to get the right time intervals
    # total_time_exp = index_time * dt_model


    # average_running_time = np.mean(t_length_each_run)
    # average_tumbling_time = np.mean(t_length_each_tumble)
    
    t_length_each_run = np.array(t_length_each_run)
    total_running_time = np.sum(t_length_each_run) 
    
    t_length_each_tumble = np.array(t_length_each_tumble)
    

    position_X = np.array(position_X)
    total_x_travelled = position_X[-1] - position_X[0]



    drift_speed = total_x_travelled / total_running_time

    #v_x for each run
    v_x_with0s = (position_X[:-1] - position_X[1:] ) / dt_model
    # v_x = (position_x_noduplicates[1:] - position_x_noduplicates[:-1]) / dt_model


    return time_points, drift_speed, position_X, v_x_with0s#, t_length_each_run, t_length_each_tumble


num_points_xt, v_d, pos_x, v_x_with0 = chemotaxis(dt_model,time_points,phi0_random,
                                                  theta0_random,randomness_diff_DR,randomness_diff_DT,randomness_arrival_rate,z_random, randomness_metnoise)

v_d = np.array([v_d]) # v_d is scalar
num_points_xt = np.array([num_points_xt]) # num_points_xt is a scalar
# v_x_with0 = np.array([v_x_with0])

result_array = np.concatenate((num_points_xt, v_d, pos_x, v_x_with0 )) 

np.savetxt('Main_output_'+str(s)+'.txt',result_array) #prints output as an array labeled by 's' in the filename
print(result_array)








