############################
#
# Ranavirus simulation.
#
# This program simulates an outbreak of ranavirus on the frogs in a pond.
# The pond is modeled as an m×n grid where each grid square contains one
# frog consisting of coordinates (0,0) to (m-1, n-1) and a state that
# indicates its infection status. With each pass (corresponding to a season
# or year) we determine for each frog its infection status based on the
# infection status of each frog in an adjacent square. Thus the frog in
# square (i, i) may be infected by any of the 8 frogs in the squares
# (i-1, i-1), (i-1, i), (i-1, 1+1), (i, i-1), (i, i+1), (i+1, i-1),
# (i+1, i), or (i+1, 1+1). (Some of these squares may not be in the pond,
# which is not infinite in extent; such squares are ignored.)
#
# Such parameters as size of pond, probability of infection, likelihood of
# death, and numbers of frogs in each state at the beginning may be set by
# the user in a block of assignments at the beginning of the program.
# Each pass of the algorithm tests each frog to see whether it changes in
# infection status due to an infected frog in an adjacent square and tests
# to determine whether an infected frog dies during the current pass.
#
# We neglect a probability Psi that a pair of adjacent frogs will interact
# because we can correct for that by adjusting the probability of infection
# between frogs in adjacent squares. 
#
# We assume that deaths of healthy frogs are inconsequential, as such
# frogs may be replaced by newborn frogs. This is because pond populations
# tend to remain somewhat stable in the absence of disease. Deaths of
# diseased frogs do not lead to replacement. Future work on this program
# may change these assumptions. We also assume that mortality is the
# same from either strain.
#
# We further assume that the probability of infection from a frog that is
# not in an adjacent square is zero; this too could be modified, though at
# a significant cost in runtime. Also, we assume that the likelihood of a
# frog with one strain of ranavirus being infected by the other strain is
# the same as for a frog with neither strain.
#
# The two strains of the ranavirus are the ulcerative (U) and the
# hemorrhagic (R); frogs with both strains are listed as Combined (C).
# At each pass, the frog in position (i, j) is tested to see whether it
# becomes infected with a strain to which it has not heretofore been
# exposed. We assume that a frog that was infected during the current
# pass must wait until the next pass to infect a neighbor.
#
# Variables beginning with "Sus_" refer to frogs in the Susceptible
# state; similarly, "Ulc_", "Hem_", "Com_", and "Dea_" denote frogs
# in the Ulcerative, Hemorrhagic, Combined, and Dead states (resp.).
############################

import random as rnd                  # Random number generator
rnd.seed(a = 28473892)                # Seed for reproducibility

# Parameter values: (change with care)

lnt = 10            # Number of columns in grid.
wid = 10            # Number of rows in grid.

Sus = lnt * wid     # Number of healthy frogs
Ulc = 0             # Number of frogs with only Ulcerative strain
Hem = 0             # Number of frogs with only Hemorrhagic strain
Com = 0             # Number of frogs with both strains
Dea = 0             # Number of dead frogs

pu = 0.65           # Probability of catching Ulcerative in any encounter
ph = 0.75           # Probability of catching Hemorrhagic in any encounter
pd = 0.250          # Probability of infected frog dying in each pass

Iters = 120         # Number of passes through the loop per pond
total_ponds = 10000 # Number of ponds to average over

# To keep running total over repeated ponds
Sus_sum = []; Ulc_sum = []; Hem_sum = []; Com_sum = []; Dea_sum = []
Sus_max = []; Ulc_max = []; Hem_max = []; Com_max = []; Dea_max = []
Sus_min = []; Ulc_min = []; Hem_min = []; Com_min = []; Dea_min = []

# Preload sums
for i in range(Iters+1):    
    Sus_sum.append(0); Sus_max.append(0); Sus_min.append(lnt * wid)
    Ulc_sum.append(0); Ulc_max.append(0); Ulc_min.append(lnt * wid)
    Hem_sum.append(0); Hem_max.append(0); Hem_min.append(lnt * wid)
    Com_sum.append(0); Com_max.append(0); Com_min.append(lnt * wid)
    Dea_sum.append(0); Dea_max.append(0); Dea_min.append(lnt * wid)
    All_infected_by = 0     # Average iteration number for no more Sus frogs
    All_dead_by = 0         # Average iteration number for no more living frogs
    All_dead_ctr = 0        # Number of ponds of all-dead frogs

# Code to make images of pond

import matplotlib.pyplot as plt
import numpy as np

# Code to construct a pond

class Frog():       # Class of frogs
    coords = [0,0]  # Position of frog in grid
    curstate="S"    # Current state of frog
    futstate="S"    #  Future state of frog
    def __init__(self):
        pass
    def showinfo(self):     # Mostly for debugging 
        print()
        print("Frog in position ", self.coords, "in state ", self.curstate)
    def set_coords(self, i, j):
        self.coords = [i, j]
    def set_state(self, newstate):      # Assign state U, H, C, or D
        self.futstate=newstate
    def update(self):               # Not until the end of the pass!
        self.curstate = self.futstate

def initialize_pond(wid, lnt):      # Create a pond with all S frogs
    pond=[]
    for k in range(lnt):
        pond.append([])
        for l in range(wid):
            newfrog = Frog()
            pond[k].append(newfrog)
            pond[k][l].set_coords(k, l)
        #next l
    #next k
    return pond

def stategrid(pond):    # Make a grid of states of frogs in pond
    '''Given an array of frog objects, creates an array of states for each frog.'''
    row = 0
    grid = []
    for i in pond:      # Create a row in grid
        grid.append([])
        for j in i:     # Enter a column entry
            grid[row].append(j.curstate)
        row += 1
    return(grid)

def showme(pond):       # Routine to draw grid of infection states
    '''Given an array of frog objects, prints a grid of states for each frog.'''
    for i in pond:
        for k in i:
            print(k.curstate, end="  ")
        print()
def find_nbrs(frog):        # List of coords of neighboring frogs
    '''Given a frog in the array, returns a list of coordinates of frogs
in squares that share an edge or corner with it.'''
    nbr_lst = []
    i = frog.coords[0]; j = frog.coords[1]
    for addr in [[i-1,j-1],[i-1,j],[i-1,j+1],[i,j-1],[i,j+1],[i+1,j-1],
                    [i+1,j],[i+1,j+1]]:
        if addr[0] >= 0 and addr[1] >= 0 and addr[0] < wid and addr[1] < lnt:
            nbr_lst.append(addr)
    return(nbr_lst)
    
def test_infect(curfrog, nbrfrog):      # Determine whether infection occurs
    '''Accepts a pair of frog objects, and tests using the random number
generator to determine whether the second frog (nbrfrog) infects
the second (curfrog).'''
    global Ulc, Com, Sus, Hem, Dea
    curstat = curfrog.futstate
    nbrstat = nbrfrog.curstate

    hemflag = 0         # To get the counters right
    if curstat == "U" or curstat == "S":              # Check for hemorrhagic
        if nbrstat == "H" or nbrstat == "C":          # Infection is possible
            # Determine if infection occurs
            test = rnd.random()         # Get random number
            if test <= ph:              # Infection with H occurs
                hemflag = 1             # So we decrement properly
                if curstat == "U":
                    curfrog.set_state("C")
                    Ulc -= 1
                    Com += 1                # Update counts of frogs in U and C
                if curstat == "S":          # Its first infection
                    curfrog.set_state("H")
                    Sus -= 1                # Update counts of frogs in S and H
                    Hem += 1
    if curstat == "H" or curstat == "S":          # Check for ulcerative
        if nbrstat == "U" or nbrstat == "C":      # Infection is possible
            # Determine if infection occurs
            test = rnd.random()         # U and H should be independent
            if test <= pu:              # Infection with U occurs

                if curstat == "H":
                    curfrog.set_state("C")
                    Hem -= 1
                    Com += 1            # Update counts of frogs in H and C
                if curstat == "S":      # Its first infection?
                    if hemflag == 1:
                        Hem -= 1
                    else:
                        Sus -= 1
                    Ulc += 1
                    curfrog.set_state("U")

# All right, here we go!

for pondcount in range(total_ponds):
    Pond = initialize_pond(wid, lnt)
    Sus = lnt * wid; Ulc = Hem = Com = Dea = 0
    Pond[wid-2][lnt-2].set_state("C")   # Assume one frog infected with both strains
    Pond[wid-2][lnt-2].update()
    Sus -= 1; Com += 1                  # Adjust counts
    Sus_sum[0] = Sus_sum[0] + lnt * wid - 1; Com_sum[0] = Com_sum[0] + 1;

    # Here is the main loop where we go through the entire pond and determine
    # whether disease transmission occurs.

    array_of_stategrids = []
    list_of_state_vecs = []
    s_flag = d_flag = 0             # Initial values to record changes

    for ctr in range(Iters):
        for i in Pond:              # Each row in Pond
            for j in i:             # Each frog in Pond[i]
                curfrog = j
                status = curfrog.curstate
                if status != "C" and status != "D":      # It can get sicker 
                    nbrs = find_nbrs(j)
                    for nbr in nbrs:     # Check every neighbor
                        r=nbr[0]; s=nbr[1]
                        nbrfrog = Pond[r][s]
                        nbrstat = nbrfrog.curstate
                        if nbrstat != "S" and nbrstat != "D":  # No need to check
                            test_infect(curfrog, nbrfrog)
        for i in Pond:              # Cleanup, update, test for death
            for j in i:
                curfrog = j
                curfrog.update()
                status = curfrog.curstate
                if status != "S" and status != "D":
                    test = rnd.random() # Check to see if the frog died.
                    if test <= pd:      # Could modify if prob of death varies between strains
                        curfrog.set_state("D")
                        Dea += 1
                        if status == "U":
                            Ulc -= 1
                        elif status == "H":
                            Hem -= 1
                        elif status == "C":
                            Com -= 1
                        curfrog.update()    # Newly infected frogs get conditions updated

        list_of_state_vecs.append((Sus, Ulc, Hem, Com, Dea))    # Record how many of each
        Sus_sum[ctr+1] += Sus; Ulc_sum[ctr+1] += Ulc; Hem_sum[ctr+1] += Hem
        Com_sum[ctr+1] += Com; Dea_sum[ctr+1] += Dea            # Total over number of ponds
        if Sus >= Sus_max[ctr+1]:
            Sus_max[ctr+1] = Sus    # Find maximal value of Sus over all ponds for this iteration
        if Sus <= Sus_min[ctr+1]:
            Sus_min[ctr+1] = Sus    # Find minimal value of Sus over all ponds for this iteration
            # if Sus == 0:
            #    print('Sus 0 in pond', ctr+1)
        if Ulc >= Ulc_max[ctr+1]:
            Ulc_max[ctr+1] = Ulc    # Find maximal value of Ulc over all ponds for this iteration
        if Ulc <= Ulc_min[ctr+1]:
            Ulc_min[ctr+1] = Ulc    # Find minimal value of Ulc over all ponds for this iteration
#            if Ulc > 0:
#               print('Ulc > 0 in iter', ctr+1, end = " ")
        if Hem >= Hem_max[ctr+1]:
            Hem_max[ctr+1] = Hem    # Find maximal value of Hem over all ponds for this iteration
        if Hem <= Hem_min[ctr+1]:
            Hem_min[ctr+1] = Hem    # Find minimal value of Hem over all ponds for this iteration
#            if Hem > 0:
#                print('Hem > 0 in iter', ctr+1, end = " ")
        if Com >= Com_max[ctr+1]:
            Com_max[ctr+1] = Com    # Find maximal value of Com over all ponds for this iteration
        if Com <= Com_min[ctr+1]:
            Com_min[ctr+1] = Com    # Find minimal value of Com over all ponds for this iteration
#            if Com > 0:
#                print('Com > 0 in iter', ctr+1, end = ' ')
        if Dea >= Dea_max[ctr+1]:
            Dea_max[ctr+1] = Dea    # Find maximal value of Dea over all ponds for this iteration
        if Dea <= Dea_min[ctr+1]:
            Dea_min[ctr+1] = Dea    # Find minimal value of Dea over all ponds for this iteration
#        print()
        
        if Sus == 0:
            if not s_flag:
    #            print ("No healthy frogs in iter # ", ctr+1)
                s_flag = 1
                # Add to figure mean iteration in which all frogs are infected
                All_infected_by += ctr + 1
    #    if Ulc == 0 and Com > 1:
    #        if not u_flag:
    #            print ("No ulcerative in iter # ", ctr)
    #            u_flag = 1
    #    if Hem == 0 and Com > 1:
    #        if not h_flag:
    #            print ("No hemorrhagic in iter # ", ctr)
    #            h_flag = 1
        if Dea == lnt * wid:
            if not d_flag:
    #            print ("No living frogs in iter # ", ctr)
                d_flag = 1
                # Add to figure mean iteration in which all frogs are dead
                All_dead_ctr += 1
                All_dead_by += ctr + 1
    #    if (ctr+1) <= 9:             # To keep copies of pond over iterations
    #        current_state_grid = stategrid(Pond)
    #        array_of_stategrids.append(current_state_grid)
    #       showme(Pond)
    #        print("Iteration number ", ctr+1, "Sus = ", Sus, "Ulc = ", Ulc, "Hem = ", Hem,
    #          "Com = ", Com, "Dea = ", Dea)
    data1=[]; data2=[]          # Make data lists of lattice coordinates
    for i in Pond:
        for j in i:
            data1.append(j.coords[0]); data2.append(j.coords[1])

Sus_list = []; Ulc_list = []; Hem_list = []; Com_list = []; Dea_list = []
for i in range(Iters+1):
    Sus_list.append(Sus_sum[i]/total_ponds)
    Ulc_list.append(Ulc_sum[i]/total_ponds)     # Compute averages over all ponds
    Hem_list.append(Hem_sum[i]/total_ponds)
    Com_list.append(Com_sum[i]/total_ponds)
    Dea_list.append(Dea_sum[i]/total_ponds)
    # This next line checks to make sure counts are correct
    # print(Sus_list[i]+Ulc_list[i]+Hem_list[i]+Com_list[i]+Dea_list[i] - lnt*wid, end=" .. ")

print("Prob of death =", pd, "using", Iters, "iterations, and", total_ponds, "ponds.")
print("\n", "Average time to all infected is ", All_infected_by/total_ponds, "iterations.")
if All_dead_ctr != 0:
    print(" Average time to all dead is ", All_dead_by/All_dead_ctr, "iterations, out of ",
          All_dead_ctr, " ponds.")
# Try to make graph of lists

x=range(ctr+2)
fig, ax = plt.subplots()
ax.set_facecolor('white')
ax.set_title("Frog Status Count vs. Time Chart", color='black')
ax.set_xlabel('Time (iterations)', color = 'black')
ax.set_ylabel('# Frogs in given status', color = 'black')
ax.plot(x, Sus_list, 'green', label="Susceptible")
ax.plot(x, Ulc_list, 'xkcd:dark yellow', label="Ulcerative")
ax.plot(x, Hem_list, 'red', label="Hemorrhagic")
ax.plot(x, Com_list, 'orange', label="Combined")
ax.plot(x, Dea_list, 'black', label="Dead")
ax.legend(loc='center right')
plt.show()



def make_pic_Sus_eb():
    '''Uses Sus_list to make a graph of the Susceptible population vs. iterations; omits initial
state. Includes error bars.'''
    fig, ax = plt.subplots()
    ax.set_facecolor('white')
    ax.plot(x[1:], Sus_list[1:], color='green')
    lower_error = []; upper_error = []
    for i in range(len(Sus_list)):
        lower_error.append(Sus_list[i] - Sus_min[i])
        upper_error.append(Sus_max[i] - Sus_list[i])
    asymmetric_error = [lower_error[1:], upper_error[1:]]
    ax.errorbar(x[1:], Sus_list[1:], color = 'green', yerr=asymmetric_error)
    plt.show()

def make_pic_Sus_tr():
    '''Uses Sus_list to make a graph of the Susceptible population vs. iterations; omits initial
state. Includes max and min values as transparent.'''
    fig, ax = plt.subplots()
    ax.set_facecolor('white')
    ax.plot(x[1:], Sus_list[1:], color='green')
    ax.fill_between(x[1:],Sus_min[1:], Sus_max[1:], alpha=0.4, color='green')
    ax.set_xlabel('Iteration Count')
    ax.set_ylabel('Number of Susceptible Frogs')
    ax.set_title('Susceptible Population vs. Iterations')
    plt.show()

def make_pic_Ulc_eb():
    '''Uses Ulc_list to make a graph of the Ulcerative population vs. iterations; omits initial state.'''
    fig, ax = plt.subplots()
    ax.set_facecolor('white')
    ax.plot(x[1:], Ulc_list[1:], color='xkcd:dark yellow')
    lower_error = []; upper_error = []
    for i in range(len(Ulc_list)):
        lower_error.append(Ulc_list[i] - Ulc_min[i])
        upper_error.append(Ulc_max[i] - Ulc_list[i])
    asymmetric_error = [lower_error[1:], upper_error[1:]]
    ax.errorbar(x[1:], Ulc_list[1:], color='xkcd:dark yellow', yerr=asymmetric_error)
    plt.show()

def make_pic_Ulc_tr():
    '''Uses Ulc_list to make a graph of the Ulcerative population vs. iterations; omits initial
state.  Includes max and min values as transparent.'''
    fig, ax = plt.subplots()
    ax.set_facecolor('white')
    ax.plot(x[1:], Ulc_list[1:], color='xkcd:dark yellow')
    ax.fill_between(x[1:],Ulc_min[1:], Ulc_max[1:], alpha=0.4, color='xkcd:dark yellow')
    ax.set_xlabel('Iteration Count')
    ax.set_ylabel('Number of Ulcerative Frogs')
    ax.set_title('Ulcerative Population vs. Iterations')
    plt.show()

def make_pic_Hem_tr():
    '''Uses Hem_list to make a graph of the Hemorrhagic population vs. iterations; omits initial
state.  Includes max and min values as transparent.'''
    fig, ax = plt.subplots()
    ax.set_facecolor('white')
    ax.plot(x[1:], Hem_list[1:], color='red')
    ax.fill_between(x[1:],Hem_min[1:], Hem_max[1:], alpha=0.4, color='red')
    ax.set_xlabel('Iteration Count')
    ax.set_ylabel('Number of Hemorrhagic Frogs')
    ax.set_title('Hemorrhagic Population vs. Iterations')
    plt.show()

def make_pic_Com_tr():
    '''Uses Com_list to make a graph of the Combine population vs. iterations; omits initial
state.  Includes max and min values as transparent.'''
    fig, ax = plt.subplots()
    ax.set_facecolor('white')
    ax.plot(x[1:], Com_list[1:], color='orange')
    ax.fill_between(x[1:],Com_min[1:], Com_max[1:], alpha=0.4, color='orange')
    ax.set_xlabel('Iteration Count')
    ax.set_ylabel('Number of Frogs With Both Strains')
    ax.set_title('Population of Frogs with Both Strains vs. Iterations')
    plt.show()

def make_pic_Dea_tr():
    '''Uses Dea_list to make a graph of the Dead population vs. iterations; omits initial
state.  Includes max and min values as transparent.'''
    fig, ax = plt.subplots()
    ax.set_facecolor('white')
    ax.plot(x[1:], Dea_list[1:], color='black')
    ax.fill_between(x[1:],Dea_min[1:], Dea_max[1:], alpha=0.4, color='gray')
    ax.set_xlabel('Iteration Count')
    ax.set_ylabel('Number of Dead Frogs')
    ax.set_title('Population of Dead Frogs vs. Iterations')
    plt.show()

def make_pic_Hem_eb():
    '''Uses Hem_list to make a graph of the Hemorrhagic population vs. iterations; omits initial state.'''
    fig, ax = plt.subplots()
    ax.set_facecolor('white')
    ax.plot(x[1:], Hem_list[1:], color='red')
    lower_error = []; upper_error = []
    for i in range(len(Hem_list)):
        lower_error.append(Hem_list[i] - Hem_min[i])
        upper_error.append(Hem_max[i] - Hem_list[i])
    asymmetric_error = [lower_error[1:], upper_error[1:]]
    ax.errorbar(x[1:], Hem_list[1:], color = 'red', yerr=asymmetric_error)
    plt.show()

def make_pic_Com():
    '''Uses Com_list to make a graph of the Combined population vs. iterations; omits initial state.'''
    fig, ax = plt.subplots()
    ax.set_facecolor('white')
    ax.plot(x[1:], Com_list[1:], color='orange')
    lower_error = []; upper_error = []
    for i in range(len(Com_list)):
        lower_error.append(Com_list[i] - Com_min[i])
        upper_error.append(Com_max[i] - Com_list[i])
    asymmetric_error = [lower_error[1:], upper_error[1:]]
    ax.errorbar(x[1:], Com_list[1:], color = 'orange', yerr=asymmetric_error)
    plt.show()

def make_pic_Dea():
    '''Uses Dea_list to make a graph of the Dead population vs. iterations; omits initial state.'''
    fig, ax = plt.subplots()
    ax.set_facecolor('white')
    ax.plot(x[1:], Dea_list[1:], color='black')
    lower_error = []; upper_error = []
    for i in range(len(Dea_list)):
        lower_error.append(Dea_list[i] - Dea_min[i])
        upper_error.append(Dea_max[i] - Dea_list[i])
    asymmetric_error = [lower_error[1:], upper_error[1:]]
    ax.errorbar(x[1:], Dea_list[1:], color = 'black', yerr=asymmetric_error)
    plt.show()

def make_pic_stategrid(stategrid, rad):     # Produce a picture of the state grid
    '''Given a rectangular array of states, from "H", "U", "S", and "D", we
create a rectangular array of colored circles in which state "H" is
colored red, "U" is colored yellow, "S" is green, and "D" is black. '''
    data1 = []; data2 = []      # Make data lists of lattice coordinates
    for i in range(len(stategrid)):
        for j in range(len(stategrid[0])):
            data1.append(1+i/8)
            data2.append(1+j/8)
    # Select colors for state; Red = 'H', yellow = 'U', green = 'S', black = 'D'
    colorslist = []
    for i in stategrid:
        for j in i:
            if j == 'S':
                colorslist.append('green')
            elif j == 'U':
                colorslist.append('yellow')
            elif j == 'H':
                colorslist.append('red')
            elif j == 'C':
                colorslist.append('orange')
            elif j == 'D':
                colorslist.append('black')
            else:
                print('We should never get here', ' Frog ', stategrid[i][j].coords)
    fig, ax = plt.subplots()
    ax.set_xlim(left=0.75, right = 1.60)
    ax.set_ylim(bottom=0.75, top = 1.60)
    ax.scatter(data1, data2, s=rad, facecolor=colorslist, edgecolor='k')
    plt.show()

