import matplotlib.pyplot as plt
import os
plt.style.use('./accretion.mplstyle')
import numpy as np
import random as rand
from astropy import constants as const
from astropy import units as u
from collections import OrderedDict
from scipy import interpolate
import time

start=time.time()

def get_elapsed_time(start=0):
    end = time.time() - start
    time_label = 'seconds'    
    if end > 3600:
        end = end/3600
        time_label = 'hours'
    elif end > 60:
        end = end/60
        time_label = 'minutes'
    return end, time_label

# Functions for integrator in 3D.

def dvidt(M_j,r_ij,r_i,r_j):
    G = 498.2338252799999 # const.G.cgs.to(u.cm**3/u.g/u.day**2).value
    dvdt=[0,0]
    dvdt[0] = -(G*M_j/r_ij**3*(r_i[0] - r_j[0]))
    dvdt[1] = -(G*M_j/r_ij**3*(r_i[1] - r_j[1]))
    return dvdt

def rmag(r_i,r_j):
    x_i=r_i[0]
    y_i=r_i[1]
    x_j=r_j[0]
    y_j=r_j[1]
    return np.sqrt((x_i-x_j)**2+(y_i-y_j)**2)

def r_s_mag(r_i,r_j):
    x_i=r_i[0]
    x_j=r_j[0]
    return x_i-x_j

def get_r_final(r,v,dt):
    x,y = np.add(r,v*dt)
    return x,y

def ttv_model(m_venus):

    b = OrderedDict((("Sun",0),("Venus",1),("Earth",2)))

    m_sun = 1.988409870698051e+33 # const.M_sun.to(u.g).value
    m_v = m_venus*1e27 #*u.g # (4.8673*10**27)*u.g
    m_e = 5.972167867791379e+27 # const.M_earth.to(u.g).value
    m = OrderedDict(((0,m_sun),(1,m_v),(2,m_e)))

    AU_cm = 14959787070000.0
    a_v = 0.723*AU_cm # const.au.to(u.cm).value
    a_e = AU_cm # const.au.to(u.cm).value
    a = OrderedDict(((1,a_v),(2,a_e)))
    a_sun = -(m[1]*a[1] + m[2]*a[2])/m_sun
    a = OrderedDict(((0,a_sun),(1,a_v),(2,a_e)))

    v_sun = -1123200.0 # -(0.13*u.m/u.s).to(u.cm/u.day).value
    v_v = 303613056000.0 # 1.18*(29.78*u.km/u.s).to(u.cm/u.day).value
    v_e = 257299200000.0 # (29.78*u.km/u.s).to(u.cm/u.day).value
    vstart = OrderedDict(((0,v_sun),(1,v_v),(2,v_e)))

    # colors = OrderedDict(((0,'white'),(1,'yellow'),(2,'blue')))
    period = OrderedDict(((0,0),(1,0.615),(2,1)))

    # Set timesteps and number of orbits here

    delta_t = 1 # in days

    orbits = 50

    t_range = np.arange(0,orbits*365,delta_t)      # 1 year timescale in steps of delta_t

    # Initialize empty position, velocity, and acceleration arrays
    Dimensions=2
    r = [[[0,0] for i in range(len(t_range))] for i in range(len(b))]
    x = [[0]*len(t_range) for i in range(len(b))]
    y = [[0]*len(t_range) for i in range(len(b))]
    v = [[[0,0] for i in range(len(t_range))] for i in range(len(b))]
    dvdt = [[[0,0] for i in range(len(t_range))] for i in range(len(b))]

    # Set initial position for each body
    for i,bdy in enumerate(b):
        r[i][0][0] = a[i]
        r[i][0][1] = 0
        x[i][0] = r[i][0][0]
        y[i][0] = r[i][0][1]
        v[i][0][0] = 0
        v[i][0][1] = vstart[i]

    # Integrate over all timesteps to get orbital parameters for each body

    steps=len(t_range) # The number of steps in the time array
        
    for t in range(steps-1): # indexing each timestep as t
        # Update the next position for each body
        for i, body in enumerate(b):
            r[i][t+1][0], r[i][t+1][1] = get_r_final(r[i][t],v[i][t],delta_t)
            x[i][t+1], y[i][t+1] = r[i][t+1][0], r[i][t+1][1]

        # r_ij
        r_01 = rmag(r[0][t+1],r[1][t+1])
        r_02 = rmag(r[0][t+1],r[2][t+1])
        r_10 = rmag(r[1][t+1],r[0][t+1])
        r_12 = rmag(r[1][t+1],r[2][t+1])
        r_20 = rmag(r[2][t+1],r[0][t+1])
        r_21 = rmag(r[2][t+1],r[1][t+1])

        # dvdt
        dvdt[0][t] = np.add(dvidt(m[1],r_01,r[0][t+1],r[1][t+1]), dvidt(m[2],r_02,r[0][t+1],r[2][t+1]))
        dvdt[1][t] = np.add(dvidt(m[0],r_10,r[1][t+1],r[0][t+1]), dvidt(m[2],r_12,r[1][t+1],r[2][t+1]))
        dvdt[2][t] = np.add(dvidt(m[0],r_20,r[2][t+1],r[0][t+1]), dvidt(m[1],r_21,r[2][t+1],r[1][t+1]))

        # v
        v[0][t+1] = np.add(v[0][t], dvdt[0][t] * delta_t)
        v[1][t+1] = np.add(v[1][t], dvdt[1][t] * delta_t)
        v[2][t+1] = np.add(v[2][t], dvdt[2][t] * delta_t)

    # Set global functions based on calculated orbital parameter arrays
    xfunc=[0 for i in b]
    yfunc=[0 for i in b]
    vfunc=[[0,0] for i in b]
    afunc=[[0,0] for i in b]
    for i,body in enumerate(b):
        xfunc[i] = interpolate.interp1d([t for t in t_range],[k for k in x[i]])
        yfunc[i] = interpolate.interp1d([t for t in t_range],[k for k in y[i]])
        for D in range(Dimensions):
            vfunc[i][D] = interpolate.interp1d([t for t in t_range],[k[D] for k in v[i]])
            afunc[i][D] = interpolate.interp1d([t for t in t_range],[k[D] for k in dvdt[i]])

    # FUNCTIONS BELOW ARE TEMPORARILY IN 2D ONLY
    def g(transitting_body,t):
        i = transitting_body
        return (xfunc[i](t)-xfunc[0](t))*(vfunc[i][0](t)-vfunc[0][0](t))

    def dgdt(transitting_body,t):
        i = transitting_body
        return (vfunc[i][0](t)-vfunc[0][0](t))**2+(xfunc[i](t)-xfunc[0](t))*(afunc[i][0](t)-afunc[0][0](t))

    def get_approx_transits(transitting_body,t_array,num_orbits):
        i = transitting_body
        trange=[int(t) for t in t_array]
        approx_transits = [0 for k in range(num_orbits)]
        orbit=0
        for t in trange:
            if yfunc[i](t)>0 and t>trange[0] and t<trange[-1]:
                if abs(xfunc[i](t)-xfunc[0](t)) < abs(xfunc[i](t-1)-xfunc[0](t-1)) and abs(xfunc[i](t)-xfunc[0](t)) < abs(xfunc[i](t+1)-xfunc[0](t+1)):
                    approx_transits[orbit] = t
                    orbit+=1
        return approx_transits

    def pinpoint_transits(transitting_body,approx_transits):
        count=0
        while max(g(transitting_body,approx_transits)) > 10**-15:
            g_array = g(transitting_body,approx_transits)
            dgdt_array = dgdt(transitting_body,approx_transits)
            approx_transits += - g_array / dgdt_array
            count += 1
            if count == 10:
                print(f'You fucked up. Count: {count}.\nBreaking While Loop Manually.')
                break
        return approx_transits

    num_orbits_earth=int(orbits/period[2])
    approx_transits_earth=get_approx_transits(2,t_range,num_orbits_earth)
    real_transits_earth=pinpoint_transits(2,approx_transits_earth)
    return real_transits_earth

# MCMC
for trial in range(3):

    # transit_file = f'/storage/home/nxt5197/work/577_exoplanets/HW10/transits{trial}.txt'
    counter_file = f'/storage/home/nxt5197/work/577_exoplanets/HW10/counter{trial}.txt'
    # if os.path.isfile(transit_file):
    #     os.remove(transit_file)
    if os.path.isfile(counter_file):
        os.remove(counter_file)


    def lnp(m,sigma,data):
        sigma2 = sigma**2
        n = len(data)
        x = ttv_model(m)
        # print(f'data: {data}')
        # print(f'm: {m}')
        # print(f'model(m): {x}')
        return - sum((data-x)**2)/2/sigma2

    actual_mass = 4.8673 # e27
    ttv_data = ttv_model(actual_mass)
    sigma = np.random.normal(0.1/60/24,10/60/24)
    error = sigma
    ttv_noisy = np.array([ttv_data[i] + np.random.normal(0,sigma) for i in range(len(ttv_data))])

    # with open(transit_file,'a') as f:
    #     f.write(f'Venus Mass: {(actual_mass*10**27):.6e}\tEarth Transit Times: {ttv_data}\n')

    number_of_steps=1000

    # plt.errorbar(range(1,1+number_of_steps),ttv_noisy/(365*0.25*np.arange(1,1+number_of_steps)**2)-1,yerr=error,ls='none',capsize=5,marker='s',color='k')
    # plt.plot(range(1,1+number_of_steps),ttv_data/(365*0.25*np.arange(1,1+number_of_steps)**2)-1)
    # plt.xlim(1,number_of_steps)
    # plt.show()

    mass_guess = 4.9 # e27
    z = mass_guess
    step_size = 0.1
    s = step_size
    # print(f'\nInitial guess:\n  mass (g)\tstep size\n {z:.4e}\t  {s:.2f}')

    zs = [0 for i in range(number_of_steps)]

    lnp0 = lnp(z,sigma,ttv_noisy)

    accepted=0
    count=0
    for step in range(number_of_steps):
        jump = s*np.random.normal(0,1) # *1e27
        new_z = z+jump
        # print(f'z: {z}\tjump: {jump}\tnew_z: {new_z}')
        lnp1 = lnp(new_z,sigma,ttv_noisy)
        # print(f'lnp0: {lnp0}\tlnp1: {lnp1}')
        prob = min(1,np.exp(lnp1-lnp0))
        # print(f'np.exp(lnp1-lnp0): {np.exp(lnp1-lnp0)}')
        # print(f'prob: {prob}')
        if prob > rand.uniform(0,1):
            z = new_z
            accepted+=1
            lnp0 = lnp1
            # print('ACCEPTED')
        # else:
            # print('NOT ACCEPTED')
        count+=1
        end, time_label = get_elapsed_time(start)
        with open(counter_file, 'a') as f:
            f.write(f'\n{step+1} of {number_of_steps} for loops completed in {end:.2f} {time_label}. Current acceptance fraction:\t{accepted/count*100:.2f}%')
        zs[step] = z
    print(f'\nTrial {trial} acceptance fraction:\t{accepted/count*100:.2f}%')

    masses=np.array(zs)

    os.chdir('/storage/home/nxt5197/work/577_exoplanets/HW10/')

    fig,ax = plt.subplots(1)
    ax.plot(range(number_of_steps),masses,color='k')
    ax.hlines(actual_mass,-1,number_of_steps+1,color='r',linestyle=':')
    ax.set_ylabel(r'mass / ($10^{27}$ g)')
    ax.set_xlim(0,number_of_steps)
    ax.set_xlabel('step number')
    plt.savefig(f'HW10_m_v_steps{trial}.png',format='png')
    plt.close()

    print(f'\nMasses: {masses}\n')

    spread = [(masses[i]-actual_mass) for i in range(len(masses))]
    # bins = np.linspace(min(spread),max(spread),int(len(spread)/5))
    histy,histx,_ = plt.hist(spread,alpha=0.75,align='left')
    plt.vlines(0,-1,histy.max()*1.02,linestyles=':',color='r')
    plt.ylim(0,histy.max()*1.01)
    plt.xlabel(r'mass / ($10^{27}$ g) - 4.8670')
    plt.ylabel('number of samples')
    plt.savefig(f'HW10_histogram{trial}.png',format='png',bbox_inches='tight')
    plt.close()