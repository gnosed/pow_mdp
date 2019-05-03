import mdptoolbox
import numpy as np
import networkx as nx
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
'''
try: 
        import queue
except ImportError:
        import Queue as queue
        '''
import Queue
from matplotlib.colors import LogNorm
import seaborn
seaborn.set(font_scale=2.3)
seaborn.set_style("whitegrid")
import sys
import datetime

now = datetime.datetime.now()
from multiprocessing import Process
import os

class State:
    def __init__(self, w_a, w_h, f_a, match):
        self.weight_a = w_a
        self.weight_h = w_h
        self.flag_a = f_a
        self.match = match

    def __hash__(self):
        return hash((self.weight_a, self.weight_h, self.flag_a, self.match))

    def __eq__(self, other):
        try:
            return (self.weight_a, self.weight_h, self.flag_a, self.match) == (other.weight_a, other.weight_h, other.flag_a, other.match)
        except:
            return False

    def __ne__(self, other):
        return not(self == other)

    def __repr__(self):
        return "(%d, %d, %s, %s)" % (self.weight_a, self.weight_h, self.flag_a, self.match)

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

# approximation taken from the paper "probability of having left behind tips"
def poblb(alpha): 
    r = (-1)*3.0069930070036804e-002 * np.power(alpha,0) +  3.4366744366776230e+000 * np.power(alpha,1) + (-1)*4.4160839160989482e+000 * np.power(alpha,2) +  1.2626262626494620e+000 * np.power(alpha,3) +  6.9930069928938809e-001 * np.power(alpha,4)  
    if r > 1: 
        r = 1
    elif r < 0:
        r = 0

    return r 

# probability of NOT having left behind tips i.e probability of the tips being choosen
def ponblb(alpha):
    return 1-poblb(alpha)

clamp = lambda n, minn, maxn: max(min(maxn, n), minn)

def rho(weight_h, alpha):
    r = ponblb(alpha) + (ponblb(clamp(1-alpha, 0, 1))*weight_h)/100
    
    if alpha == 1:
        r = ponblb(alpha) + (ponblb(0.01)*weight_h)/100

    return clamp(r, 0, 1)

def success(weight_a, weight_h, alpha):
    if weight_a > 0 and weight_h > 0:
        r = weight_a/(weight_a + (weight_h*np.exp((-1)*alpha*(abs(weight_a - weight_h)))))
    else: 
        r = 0

    return clamp(r, 0, 1)

def optimal_strategy(p, k, stale, double_spend_value, max_blocks, gamma, cutoff, m_cost, alpha):
    """
    p: attacker hashrate as fraction of total hashrate
    stale_rate: rate of stale blocks in honest network
    double_spend_value: the value that the attacker gains from a successful double spend, 1 = block reward
    max_blocks: maximum number of total mined blocks considered
    gamma: fraction of honest nodes that a block from the attacker reaches before a block from the honest network (when matching)
    cutoff: maximum length of either chain (needed for finite MDP)
    m_cost: cost of mining (per state transition)
    lam: hashrate of eclipsed miner

    implicitly: 
        q = (1 - p - lam) hashrate of honest network
        q*(1-stale) probability that the honest chain grows (in each step)
        q*stale probability that state stays the same
    """
    lam=0
    states = {}
    states_inverted = {}
    q = 1.0-p-lam
    # randomness rho, to be replaced at the end by the appropriate function 
    #r = rho(w_h, alpha)
    #s = p + (1-p)*r 
    #success = success(w_a, w_h, alpha)
        
    flags_attack = ["inactive", "validated", "active"]
    match_cases = ["honest", "malicious", "both"]

    # construct states
    states_counter = 0
    for w_a in xrange(cutoff + 1):
        for w_h in xrange(cutoff + 1):
            for f_a in flags_attack:
                for match in match_cases:
                    state = State(w_a, w_h, f_a, match)
                    states[states_counter] = state
                    states_inverted[state] = states_counter
                    #print "state: ", state 
                    #print "states_counter: ", states_counter
                    #print "states_inverted[state]: ", states_inverted[state]
                    states_counter += 1
                    

    # exit state
    exit_idx = states_counter
    states_counter += 1

    # build transition matrices
    P_hreset    = np.zeros(shape=(states_counter, states_counter))
    P_mreset    = np.zeros(shape=(states_counter, states_counter))
    P_spam      = np.zeros(shape=(states_counter, states_counter))
    P_construct = np.zeros(shape=(states_counter, states_counter))
    P_wait      = np.zeros(shape=(states_counter, states_counter))
    P_exit      = np.zeros(shape=(states_counter, states_counter))
    
    # build reward matrices
    R_hreset    = np.zeros(shape=(states_counter, states_counter))
    R_mreset    = np.zeros(shape=(states_counter, states_counter))
    R_spam      = np.zeros(shape=(states_counter, states_counter))
    R_construct = np.zeros(shape=(states_counter, states_counter))
    R_wait      = np.zeros(shape=(states_counter, states_counter))
    R_exit      = np.zeros(shape=(states_counter, states_counter))

    # never leave exit state
    P_hreset[exit_idx, exit_idx] = 1
    P_mreset[exit_idx, exit_idx] = 1
    P_spam[exit_idx, exit_idx] = 1
    P_construct[exit_idx, exit_idx] = 1
    P_wait[exit_idx, exit_idx] = 1
    P_exit[exit_idx, exit_idx] = 1

    R_exit[exit_idx, exit_idx] = p - m_cost

    # needed for stochastic matrix, not sure if there is a better way to do this
    np.fill_diagonal(P_hreset,1)
    np.fill_diagonal(P_mreset,1)
    np.fill_diagonal(P_spam,1)
    np.fill_diagonal(P_construct,1)
    np.fill_diagonal(P_wait,1)
    np.fill_diagonal(P_exit,1)

    #print P_hreset
    #print "exit_idx: ", exit_idx
    #print states_inverted[State(20,20,"active","both")]
    #print states_inverted
    for state_idx, state in states.iteritems():
        #print "state: ", state 
        #print "states_counter: ", states_counter
        #print "states_inverted[state]: ", states_inverted[state]
        w_a = state.weight_a
        w_h = state.weight_h
        f_a = state.flag_a
        match = state.match

        r = rho(w_h, alpha)
        #r = 0.3
        s = p + (1-p)*r 
        succ = success(w_a, w_h, alpha)
        #succ = 0.6
       
        if w_a == cutoff or w_h == cutoff:
            continue

        # honest reset
        # needed for stochastic matrix, not sure if there is a better way to do this
        P_hreset[state_idx, state_idx] = 0
        R_hreset[state_idx, state_idx] = -100

        P_hreset[state_idx, states_inverted[State(0, 1, "inactive", "honest")]] = p
        R_hreset[state_idx, states_inverted[State(0, 1, "inactive", "honest")]] = 0-m_cost

        ## miner doesn't mine the tx
        P_hreset[state_idx, states_inverted[State(0, 0, "inactive", "honest")]] = (1-p)
        R_hreset[state_idx, states_inverted[State(0, 0, "inactive", "honest")]] = 0-m_cost

        # malicious reset
        # needed for stochastic matrix, not sure if there is a better way to do this
        P_mreset[state_idx, state_idx] = 0
        R_mreset[state_idx, state_idx] = -100

        P_mreset[state_idx, states_inverted[State(1, 0, "inactive", "malicious")]] = p
        R_mreset[state_idx, states_inverted[State(1, 0, "inactive", "malicious")]] = 0-m_cost
        ## miner doesn't mine the tx
        P_mreset[state_idx, states_inverted[State(0, 0, "inactive", "malicious")]] = (1-p)
        R_mreset[state_idx, states_inverted[State(0, 0, "inactive", "malicious")]] = 0-m_cost
        
        # spam
        if w_a == 0 and f_a == "inactive" and match == "honest":
            # needed for stochastic matrix, not sure if there is a better way to do this
            P_spam[state_idx, state_idx] = 0
            R_spam[state_idx, state_idx] = -100

            if w_h-1 >= k:
                P_spam[state_idx, states_inverted[State(0, w_h+1, "validated", "honest")]] = s
                R_spam[state_idx, states_inverted[State(0, w_h+1, "validated", "honest")]] = 0-m_cost

                P_spam[state_idx, states_inverted[State(0, w_h, "inactive", "honest")]] = (1-s)
                R_spam[state_idx, states_inverted[State(0, w_h, "inactive", "honest")]] = 0-m_cost

            else:
                P_spam[state_idx, states_inverted[State(0, w_h+1, "inactive", "honest")]] = s 
                R_spam[state_idx, states_inverted[State(0, w_h+1, "inactive", "honest")]] = 0-m_cost

                P_spam[state_idx, states_inverted[State(0, w_h, "inactive", "honest")]] = (1-s)
                R_spam[state_idx, states_inverted[State(0, w_h, "inactive", "honest")]] = 0-m_cost

        if w_h == 0 and f_a == "inactive" and match == "malicious":
            # needed for stochastic matrix, not sure if there is a better way to do this
            P_spam[state_idx, state_idx] = 0
            R_spam[state_idx, state_idx] = -100

            P_spam[state_idx, states_inverted[State(w_a, 1, "inactive", "both")]] = p
            R_spam[state_idx, states_inverted[State(w_a, 1, "inactive", "both")]] = 0-m_cost

            P_spam[state_idx, states_inverted[State(w_a, 0, "inactive", "honest")]] = (1-p)
            R_spam[state_idx, states_inverted[State(w_a, 0, "inactive", "honest")]] = 0-m_cost

        if f_a == "inactive" and match == "both":
            # needed for stochastic matrix, not sure if there is a better way to do this
            P_spam[state_idx, state_idx] = 0
            R_spam[state_idx, state_idx] = -100

            if w_h-1 >= k:
                P_spam[state_idx, states_inverted[State(w_a, w_h+1, "validated", "both")]] = s
                R_spam[state_idx, states_inverted[State(w_a, w_h+1, "validated", "both")]] = 0-m_cost

                P_spam[state_idx, states_inverted[State(w_a, w_h, "inactive", "both")]] = (1-s)
                R_spam[state_idx, states_inverted[State(w_a, w_h, "inactive", "both")]] = 0-m_cost
            else:
                P_spam[state_idx, states_inverted[State(w_a, w_h+1, "inactive", "both")]] = s
                R_spam[state_idx, states_inverted[State(w_a, w_h+1, "inactive", "both")]] = 0-m_cost

                P_spam[state_idx, states_inverted[State(w_a, w_h, "inactive", "both")]] = (1-s)
                R_spam[state_idx, states_inverted[State(w_a, w_h, "inactive", "both")]] = 0-m_cost
 
        # construct 
        if w_a == 0 and f_a == "inactive" and match == "honest":
            # needed for stochastic matrix, not sure if there is a better way to do this
            P_construct[state_idx, state_idx] = 0
            R_construct[state_idx, state_idx] = -100

            if w_h-1 >= k:
                P_construct[state_idx, states_inverted[State(1, w_h, "inactive", "both")]] = p
                R_construct[state_idx, states_inverted[State(1, w_h, "inactive", "both")]] = 0-m_cost

                P_construct[state_idx, states_inverted[State(0, w_h+1, "validated", "honest")]] = (1-p)*r
                R_construct[state_idx, states_inverted[State(0, w_h+1, "validated", "honest")]] = 0-m_cost

                P_construct[state_idx, states_inverted[State(0, w_h, "inactive", "honest")]] = (1-p)*(1-r)
                R_construct[state_idx, states_inverted[State(0, w_h, "inactive", "honest")]] = 0-m_cost
            else:
                P_construct[state_idx, states_inverted[State(1, w_h, "inactive", "both")]] = p
                R_construct[state_idx, states_inverted[State(1, w_h, "inactive", "both")]] = 0-m_cost

                P_construct[state_idx, states_inverted[State(0, w_h+1, "inactive", "honest")]] = (1-p)*r
                R_construct[state_idx, states_inverted[State(0, w_h+1, "inactive", "honest")]] = 0-m_cost

                P_construct[state_idx, states_inverted[State(0, w_h, "inactive", "honest")]] = (1-p)*(1-r)
                R_construct[state_idx, states_inverted[State(0, w_h, "inactive", "honest")]] = 0-m_cost
                
        if w_a == 0 and f_a == "validated" and match == "honest":
            # needed for stochastic matrix, not sure if there is a better way to do this
            P_construct[state_idx, state_idx] = 0
            R_construct[state_idx, state_idx] = -100

            P_construct[state_idx, states_inverted[State(1, w_h, "validated", "both")]] = p
            R_construct[state_idx, states_inverted[State(1, w_h, "validated", "both")]] = 0-m_cost

            P_construct[state_idx, states_inverted[State(0, w_h+1, "validated", "honest")]] = (1-p)*r
            R_construct[state_idx, states_inverted[State(0, w_h+1, "validated", "honest")]] = 0-m_cost

            P_construct[state_idx, states_inverted[State(0, w_h, "validated", "honest")]] = (1-p)*(1-r)
            R_construct[state_idx, states_inverted[State(0, w_h, "validated", "honest")]] = 0-m_cost

        if w_h == 0 and f_a == "inactive" and match == "malicious":
            # needed for stochastic matrix, not sure if there is a better way to do this
            P_construct[state_idx, state_idx] = 0
            R_construct[state_idx, state_idx] = -100

            P_construct[state_idx, states_inverted[State(w_a+1, 0, "inactive", "malicious")]] = p 
            R_construct[state_idx, states_inverted[State(w_a+1, 0, "inactive", "malicious")]] = 0-m_cost

            P_construct[state_idx, states_inverted[State(w_a, 0, "inactive", "malicious")]] = (1-p) 
            R_construct[state_idx, states_inverted[State(w_a, 0, "inactive", "malicious")]] = 0-m_cost

        if f_a == "inactive" and match == "both":
            # needed for stochastic matrix, not sure if there is a better way to do this
            P_construct[state_idx, state_idx] = 0
            R_construct[state_idx, state_idx] = -100

            if w_h-1 >= k:
                P_construct[state_idx, states_inverted[State(w_a+1, w_h, "inactive", "both")]] = p 
                R_construct[state_idx, states_inverted[State(w_a+1, w_h, "inactive", "both")]] = 0-m_cost

                P_construct[state_idx, states_inverted[State(w_a, w_h+1, "validated", "both")]] = (1-p)*r 
                R_construct[state_idx, states_inverted[State(w_a, w_h+1, "validated", "both")]] = 0-m_cost

                P_construct[state_idx, states_inverted[State(w_a, w_h, "inactive", "both")]] = (1-p)*(1-r) 
                R_construct[state_idx, states_inverted[State(w_a, w_h, "inactive", "both")]] = 0-m_cost

            else:
                P_construct[state_idx, states_inverted[State(w_a+1, w_h, "inactive", "both")]] = p 
                R_construct[state_idx, states_inverted[State(w_a+1, w_h, "inactive", "both")]] = 0-m_cost

                P_construct[state_idx, states_inverted[State(w_a, w_h+1, "inactive", "both")]] = (1-p)*r 
                R_construct[state_idx, states_inverted[State(w_a, w_h+1, "inactive", "both")]] = 0-m_cost

                P_construct[state_idx, states_inverted[State(w_a, w_h, "inactive", "both")]] = (1-p)*(1-r) 
                R_construct[state_idx, states_inverted[State(w_a, w_h, "inactive", "both")]] = 0-m_cost

        if f_a == "validated" and match == "both":
            # needed for stochastic matrix, not sure if there is a better way to do this
            P_construct[state_idx, state_idx] = 0
            R_construct[state_idx, state_idx] = -100

            if w_a-1 > w_h:
                P_construct[state_idx, states_inverted[State(w_a+1, w_h, "active", "both")]] = p 
                R_construct[state_idx, states_inverted[State(w_a+1, w_h, "active", "both")]] = 0-m_cost

                P_construct[state_idx, states_inverted[State(w_a, w_h+1, "validated", "both")]] = (1-p)*r 
                R_construct[state_idx, states_inverted[State(w_a, w_h+1, "validated", "both")]] = 0-m_cost

                P_construct[state_idx, states_inverted[State(w_a, w_h, "validated", "both")]] = (1-p)*(1-r) 
                R_construct[state_idx, states_inverted[State(w_a, w_h, "validated", "both")]] = 0-m_cost

            else:
                P_construct[state_idx, states_inverted[State(w_a+1, w_h, "validated", "both")]] = p 
                R_construct[state_idx, states_inverted[State(w_a+1, w_h, "validated", "both")]] = 0-m_cost

                P_construct[state_idx, states_inverted[State(w_a, w_h+1, "validated", "both")]] = (1-p)*r 
                R_construct[state_idx, states_inverted[State(w_a, w_h+1, "validated", "both")]] = 0-m_cost

                P_construct[state_idx, states_inverted[State(w_a, w_h, "validated", "both")]] = (1-p)*(1-r) 
                R_construct[state_idx, states_inverted[State(w_a, w_h, "validated", "both")]] = 0-m_cost

        # wait 
        if w_a == 0 and f_a == "inactive" and match == "honest":
            # needed for stochastic matrix, not sure if there is a better way to do this
            P_wait[state_idx, state_idx] = 0
            R_wait[state_idx, state_idx] = -100

            if w_h-1 >= k:
                P_wait[state_idx, states_inverted[State(0, w_h+1, "validated", "honest")]] = (1-p)*r
                R_wait[state_idx, states_inverted[State(0, w_h+1, "validated", "honest")]] = 0 

                P_wait[state_idx, states_inverted[State(0, w_h, "inactive", "honest")]] = 1-((1-p)*r)
                R_wait[state_idx, states_inverted[State(0, w_h, "inactive", "honest")]] = 0

            else:
                P_wait[state_idx, states_inverted[State(0, w_h+1, "inactive", "honest")]] = (1-p)*r
                R_wait[state_idx, states_inverted[State(0, w_h+1, "inactive", "honest")]] = 0 

                P_wait[state_idx, states_inverted[State(0, w_h, "inactive", "honest")]] = 1-((1-p)*r)
                R_wait[state_idx, states_inverted[State(0, w_h, "inactive", "honest")]] = 0

        if f_a == "inactive" and match == "both":
            # needed for stochastic matrix, not sure if there is a better way to do this
            P_wait[state_idx, state_idx] = 0
            R_wait[state_idx, state_idx] = -100

            if w_h-1 >= k:
                P_wait[state_idx, states_inverted[State(w_a, w_h+1, "validated", "both")]] = (1-p)*r
                R_wait[state_idx, states_inverted[State(w_a, w_h+1, "validated", "both")]] = 0 

                P_wait[state_idx, states_inverted[State(w_a, w_h, "inactive", "both")]] = 1-((1-p)*r)
                R_wait[state_idx, states_inverted[State(w_a, w_h, "inactive", "both")]] = 0

            else:
                P_wait[state_idx, states_inverted[State(w_a, w_h+1, "inactive", "both")]] = (1-p)*r
                R_wait[state_idx, states_inverted[State(w_a, w_h+1, "inactive", "both")]] = 0 

                P_wait[state_idx, states_inverted[State(w_a, w_h, "inactive", "both")]] = 1-((1-p)*r)
                R_wait[state_idx, states_inverted[State(w_a, w_h, "inactive", "both")]] = 0

        if f_a == "validated" and match == "both":
            # needed for stochastic matrix, not sure if there is a better way to do this
            P_wait[state_idx, state_idx] = 0
            R_wait[state_idx, state_idx] = -100

            P_wait[state_idx, states_inverted[State(w_a, w_h+1, "validated", "both")]] = (1-p)*r
            R_wait[state_idx, states_inverted[State(w_a, w_h+1, "validated", "both")]] = 0 

            P_wait[state_idx, states_inverted[State(w_a, w_h, "validated", "both")]] = 1-((1-p)*r)
            R_wait[state_idx, states_inverted[State(w_a, w_h, "validated", "both")]] = 0

        if f_a == "active" and match == "both":
            # needed for stochastic matrix, not sure if there is a better way to do this
            P_wait[state_idx, state_idx] = 0
            R_wait[state_idx, state_idx] = -100

            if w_h-1 >= w_a:
                P_wait[state_idx, states_inverted[State(w_a, w_h+1, "validated", "both")]] = (1-p)*r
                R_wait[state_idx, states_inverted[State(w_a, w_h+1, "validated", "both")]] = 0 

                P_wait[state_idx, states_inverted[State(w_a, w_h, "validated", "both")]] = 1-((1-p)*r)
                R_wait[state_idx, states_inverted[State(w_a, w_h, "validated", "both")]] = 0

            else:
                P_wait[state_idx, states_inverted[State(w_a, w_h+1, "active", "both")]] = (1-p)*r
                R_wait[state_idx, states_inverted[State(w_a, w_h+1, "active", "both")]] = 0 

                P_wait[state_idx, states_inverted[State(w_a, w_h, "active", "both")]] = 1-((1-p)*r)
                R_wait[state_idx, states_inverted[State(w_a, w_h, "active", "both")]] = 0

        # exit
        if f_a == "active" and match == "both":
            # needed for stochastic matrix, not sure if there is a better way to do this
            P_exit[state_idx, state_idx] = 0
            R_exit[state_idx, state_idx] = -100

            P_exit[state_idx, exit_idx] = succ
            R_exit[state_idx, exit_idx] = double_spend_value - m_cost
            
            P_exit[state_idx, states_inverted[State(w_a, w_h, "active", "both")]] = 1-succ
            R_exit[state_idx, states_inverted[State(w_a, w_h, "active", "both")]] = 0

    #print P_hreset 
    P = [P_hreset, P_mreset, P_spam, P_construct, P_wait, P_exit]
    R = [R_hreset, R_mreset, R_spam, R_construct, R_wait, R_exit]
    for i,p in enumerate(P):
        try:
            mdptoolbox.util.checkSquareStochastic(p)
        except:
            print ("not stochastic:", i)
            #for l in p:
                #print l
    #mdp = mdptoolbox.mdp.FiniteHorizon(P, R, 0.999, max_blocks)
    #mdp = mdptoolbox.mdp.ValueIteration(P, R, 0.999)
    #mdp = mdptoolbox.mdp.QLearning(P, R, 0.999)
    mdp = mdptoolbox.mdp.PolicyIteration(P, R, 0.999)
    #mdp.setVerbose()
    mdp.run()
    return mdp, states


def state_graph(states, transitions, policy):
    policy_colors = ["blue", "red", "grey", "yellow", "green", "magenta"]
    G = nx.DiGraph()
    q = Queue.Queue()
    visited = [False]*len(states)
    visited[0] = True
    q.put(0)
    while not q.empty():
        state_idx = q.get()
        pol = policy[state_idx]
        G.add_node(states[state_idx], color=policy_colors[pol], style="filled")
        for i, p in enumerate(transitions[pol][state_idx]):
            if p > 0:
                if i == len(states):
                    G.add_edge(states[state_idx], "exit", label=p)
                else:
                    G.add_edge(states[state_idx], states[i], label=p)
                    if not visited[i]:
                        q.put(i)
                        visited[i] = True
    return G


def state_table(states, transitions, policy, cutoff):
    policy_letter = ["hr", "mr", "s", "c", "e"]
    q = Queue.Queue()
    table = [[['*']*3]*cutoff]*cutoff
    visited = [False]*len(states)
    visited[0] = True
    q.put(0)
    while not q.empty():
        state_idx = q.get()
        pol = policy[state_idx]
        state = states[state_idx]
        if state.match == "irrelevant":
            fork_idx = 0
        elif state.match == "relevant":
            fork_idx = 1
        elif state.match == "active":
            fork_idx = 2
        else:
            raise ValueError('Invalid fork label')
        table[state.weight_a][state.weight_h][fork_idx] = policy_letter[pol]
        for i, p in enumerate(transitions[pol][state_idx]):
            if p > 0:
                if i == len(states):
                    pass
                else:
                    if not visited[i]:
                        q.put(i)
                        visited[i] = True
    return table

def print_table(table):
    l = len(table)
    print (r"\begin{tabular}{@{}c|"+ l*'c' +r"@{}}")
    print (r"\toprule")
    print ('& ' + ' & '.join(str(x) for x in range(l)) + r'\\')
    for idx, line in enumerate(table):
        print (str(idx) + '& ' +  ' & '.join([''.join(x) for x in line]) + r'\\')
        if idx < l-1:
            print (r'\midrule')
    print (r"\bottomrule")
    print (r"\end{tabular}")

def hashrate_k_plot(stale, gamma, cost, cutoff=20, als=0):
    #ps = np.arange(0.025, 0.5, 0.025)
    #ks = np.arange(0, 13, 1)
    ps = np.arange(0.1, 0.5, 0.1)
    ks = np.arange(0, 13, 1)
    #als = np.arange(0, 1, 0.1) 
    '''
    ps = np.arange(0.1, 0.3, 0.1)
    ks = np.arange(1, 2, 1)
    als = np.arange(0, 0.1, 0.1) 
    '''
    dsi = np.zeros([len(ps), len(ks), len(als)])
    ds = np.zeros([len(ps), len(ks)])
    ds_pa = np.zeros([len(ps), len(als)])
    ds_p = np.zeros([len(ps)])
    ds_a = np.zeros([len(als)])
    ds_k = np.zeros([len(ks)])
    #ds_2 = np.zeros([len(ps), len(als)])
    #ds_3 = np.z#eros([len(ks), len(als)])
    max_val = 1000000000
    #max_val = 10000
    #max_val = 122 
    eps = 0.1
    for alpha_idx, alpha in enumerate(als):
        for p_idx, p in enumerate(ps):
            for k_idx, k in enumerate(ks):
                m_cost = cost*p
                double_spend_value = max_val/2
                if k_idx > 0 and ds[p_idx, k_idx-1] > max_val - eps:
                    double_spend_value = max_val
                last_value = 0
                diff = max_val/2
                lower = 0
                upper = max_val
                while diff > eps:
                    print ("mp_a: ", p,"k: ", k,"dv: ", double_spend_value, "alpha: ", alpha, "pid: ", os.getpid())
                    mdp,states = optimal_strategy(p, k, stale, double_spend_value, None, gamma, cutoff, m_cost, alpha)
                    G = state_graph(states, mdp.P, mdp.policy)
                    diff = abs(last_value - double_spend_value)
                    if G.has_node("exit"):
                        print ("exit")
                        last_value = double_spend_value
                        upper = double_spend_value
                        double_spend_value -= (double_spend_value - lower)/2.0
                    else:
                        last_value = double_spend_value
                        lower = double_spend_value
                        double_spend_value += (upper - double_spend_value)/2.0
                ds[p_idx, k_idx] = last_value
                dsi[p_idx, k_idx, alpha_idx] = last_value
                ds_pa[p_idx, alpha_idx] = last_value
                ds_p[p_idx] = last_value
                ds_a[alpha_idx] = last_value
                ds_k[k_idx] = last_value
                #ds_2[p_idx, alpha_idx] = last_value
                #ds_3[k_idx, alpha_idx] = last_value
     #np.save("hashrate_k_double_spend_co%dg%.2fs%.4fc%.2f%s.npy" % (cutoff, gamma, stale, cost, now.strftime("%Y-%m-%d_%H:%M")), ds)
    np.save("%d_double_spend_pk_cutoff%dcost%.2f-%s.npy" % (os.getpid(), cutoff, cost, now.strftime("%Y-%m-%d_%H:%M")), ds)
    np.save("%d_double_spend_pkalpha_cutoff%dcost%.2f-%s.npy" % (os.getpid(), cutoff, cost, now.strftime("%Y-%m-%d_%H:%M")), dsi)
    np.save("%d_double_spend_p_cutoff%dcost%.2f-%s.npy" % (os.getpid(), cutoff, cost, now.strftime("%Y-%m-%d_%H:%M")), ds_p)
    np.save("%d_double_spend_a_cutoff%dcost%.2f-%s.npy" % (os.getpid(), cutoff, cost, now.strftime("%Y-%m-%d_%H:%M")), ds_a)
    np.save("%d_double_spend_k_cutoff%dcost%.2f-%s.npy" % (os.getpid(), cutoff, cost, now.strftime("%Y-%m-%d_%H:%M")), ds_k)
    plt.pcolor(ps, als, ds_pa.T, norm=LogNorm(vmin=ds_pa.min(), vmax=ds_pa.max()))
    cbar = plt.colorbar()
    cbar.set_label("double spend value")
    plt.ylabel("k")
    plt.xlabel("p")
    plt.savefig("double_spend_pk_cutoff%dcost%.2f-%s.png" % (cutoff, cost, now.strftime("%Y-%m-%d_%H:%M")))
    plt.close()
    
    '''
    #print ds
    #print dsi 
    #print ds_pa
    plt.plot(ps, norm=LogNorm(vmin=ds_pa.min(), vmax=ds_pa.max()))
    plt.ylabel("double spending value")
    plt.xlabel("mining power")
    plt.savefig("double_spend_pdv_cutoff%dcost%.2f-%s.png" % (cutoff, cost, now.strftime("%Y-%m-%d_%H:%M")))
    #plt.savefig("hashrate_k_double_spend_co%dg%.2fs%.4fc%.2f%s.png" % (cutoff, gamma, stale, cost, now.strftime("%Y-%m-%d_%H:%M")))
    plt.close()
    '''
def hashrate_lam_plot(stale, gamma, cost, cutoff=20, k=6):
    ps = np.arange(0.025, 0.5, 0.025)
    lams = np.arange(0.0, 0.5, 0.025)
    ds = np.zeros([len(ps), len(lams)])
    max_val = 1000000000
    eps = 0.1
    for p_idx, p in enumerate(ps):
        for lam_idx, lam in enumerate(lams):
            m_cost = cost*p
            double_spend_value = max_val/2
            last_value = 0
            diff = max_val/2
            lower = 0
            upper = max_val
            while diff > eps:
                print (p, lam, double_spend_value)
                mdp,states = optimal_strategy(p, k, stale, double_spend_value, None, gamma,cutoff, m_cost, lam=lam)
                G = state_graph(states, mdp.P, mdp.policy)
                diff = abs(last_value - double_spend_value)
                if G.has_node("exit"):
                    print ("exit")
                    last_value = double_spend_value
                    upper = double_spend_value
                    double_spend_value -= (double_spend_value - lower)/2.0
                else:
                    last_value = double_spend_value
                    lower = double_spend_value
                    double_spend_value += (upper - double_spend_value)/2.0
            ds[p_idx, lam_idx] = last_value
    np.save("hashrate_om_double_spend_co%dk%dg%.2fs%.4fc%.2f.npy" % (cutoff,k, gamma, stale, cost), ds)
    plt.pcolor(ps, lams, ds.T, norm=LogNorm(vmin=ds.min(), vmax=ds.max()))
    cbar = plt.colorbar()
    cbar.set_label("double spend value")
    plt.ylabel("Eclipsed node hashrate $\omega$")
    plt.xlabel(r"Adversary hashrate $\alpha$")
    fig = plt.gcf()
    fig.tight_layout()
    fig.set_size_inches(10,7)
    plt.savefig("hashrate_om_double_spend_co%dk%dg%.2fs%.4fc%.2f.eps" % (cutoff,k, gamma, stale, cost))
    plt.close()

                        
def markov_chain(states, transitions, policy):
    import pykov
    T = pykov.Chain()
    q = Queue.Queue()
    visited = [False]*len(states)
    visited[0] = True
    q.put(0)
    start = pykov.Vector({states[0]:1})
    while not q.empty():
        state_idx = q.get()
        pol = policy[state_idx]
        for i, p in enumerate(transitions[pol][state_idx]):
            if p > 0:
                if i == len(states):
                    T[(states[state_idx], "exit")] = p
                    T[("exit", "exit")] = 1
                else:
                    T[(states[state_idx], states[i])] = p
                    if not visited[i]:
                        q.put(i)
                        visited[i] = True
    return T, start

def markov_matrix(transitions, policy):
    n = len(policy)
    P = np.zeros((n, n))
    q = Queue.Queue()
    visited = [False]*n
    visited[0] = True
    q.put(0)
    while not q.empty():
        state_idx = q.get()
        pol = policy[state_idx]
        for i, p in enumerate(transitions[pol][state_idx]):
            if p > 0:
                P[state_idx, i] = p
                if not visited[i]:
                    q.put(i)
                    visited[i] = True
    return P

def exp_blocks_needed(p, k, stale, gamma, double_spend_value, m_cost, cutoff=20):
    mdp, states = optimal_strategy(p, k, stale, double_spend_value, None, gamma, cutoff, m_cost)
    P = markov_matrix(mdp.P, mdp.policy)
    l = len(mdp.policy)
    Q = P[0:l-1,0:l-1]
    I = np.eye(l-1)
    N = np.linalg.inv(I - Q)
    ones = np.ones((l-1,1))
    t = N.dot(ones)
    return t[0]


def main():
    log_a00 = np.loadtxt(open("/home/gno/Dropbox/eth/bathesis/pow_mdp/mdp_results/log_simulation00.csv", "rb"), delimiter=",", skiprows=1)
    log_a02 = np.loadtxt(open("/home/gno/Dropbox/eth/bathesis/pow_mdp/mdp_results/log_simulation02.csv", "rb"), delimiter=",", skiprows=1)
    log_a04 = np.loadtxt(open("/home/gno/Dropbox/eth/bathesis/pow_mdp/mdp_results/log_simulation04.csv", "rb"), delimiter=",", skiprows=1)
    log_a06 = np.loadtxt(open("/home/gno/Dropbox/eth/bathesis/pow_mdp/mdp_results/log_simulation06.csv", "rb"), delimiter=",", skiprows=1)
    log_a08 = np.loadtxt(open("/home/gno/Dropbox/eth/bathesis/pow_mdp/mdp_results/log_simulation08.csv", "rb"), delimiter=",", skiprows=1)

    mp_a00 = log_a00[:,0]
    dv_a00 = log_a00[:,2]
    #
    mp_a02 = log_a02[:,0]
    dv_a02 = log_a02[:,2]
    #
    mp_a04 = log_a04[:,0]
    dv_a04 = log_a04[:,2]
    #
    mp_a06 = log_a06[:,0]
    dv_a06 = log_a06[:,2]
    #
    mp_a08 = log_a08[:,0]
    dv_a08 = log_a08[:,2]
    
    y = dv_a00[:105]
    x = np.arange(0, 105, 1)
    plt.plot(x, y)
    plt.show()
    '''
    l = len(sys.argv)
    if l >=3:
        cost = float(sys.argv[1])
        gamma = float(sys.argv[2])
    else:
        print ("Not enough arguments")
        print ("Usage: %s <gamma> <cost>" %sys.argv[0])
        return
    stale = 0.0041
    k = 6
    cutoff = 20
    '''
    '''
    sa = np.arange(0,1.1,0.1)
    for ax, a in enumerate(sa):
    #    print a, ponblb(a), poblb(a)
        for i in xrange(0,200):
            print i, a, rho(i,a)
    '''
    #hashrate_k_plot(stale, gamma, cost, cutoff) 
    '''
    als = np.arange(0, 0.2, 0.1) 
    p1 = Process(target=hashrate_k_plot, args=(stale, gamma, cost, cutoff, als))

    als = np.arange(0.2, 0.4, 0.1) 
    p2 = Process(target=hashrate_k_plot, args=(stale, gamma, cost, cutoff, als))

    als = np.arange(0.4, 0.6, 0.1) 
    p3 = Process(target=hashrate_k_plot, args=(stale, gamma, cost, cutoff, als))

    als = np.arange(0.6, 0.8, 0.1) 
    p4 = Process(target=hashrate_k_plot, args=(stale, gamma, cost, cutoff, als))

    als = np.arange(0.8, 1.01, 0.1) 
    p5 = Process(target=hashrate_k_plot, args=(stale, gamma, cost, cutoff, als))
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    '''

if __name__=="__main__":
    main()
