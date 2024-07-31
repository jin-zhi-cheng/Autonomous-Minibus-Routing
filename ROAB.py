import copy
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import rcParams
import seaborn as sns
import networkx as nx
import numpy as np
import numpy.random as rnd
import time
import math
from scipy.stats import norm
import random
from alns import ALNS, State
from alns.accept import RecordToRecordTravel
from alns.select import RouletteWheel
from alns.stop import MaxRuntime
from alns.accept import *
from alns.select import *
from alns.stop import *
import os
import re
from scipy.stats import lognorm
import csv

class CpdptwState(State):

    def __init__(self, routes, unassigned, penalty, demand, distance, risk):
        self.routes = routes
        self.unassigned = unassigned if unassigned is not None else []
        self.penalty = penalty if penalty is not None else []
        self.demand = demand
        self.distance = distance
        self.risk = risk

    def copy(self):
        return CpdptwState(copy.deepcopy(self.routes), self.unassigned.copy(), self.penalty.copy(), \
                           self.demand.copy(), self.distance.copy(), self.risk.copy())

    def objective(self):
        dist = 0
        cus_time = 0
        bus_num = busnum_check(self.routes)
        pena_num = len(self.penalty)
        for route in self.routes:
            tour = [0] + route + [0]
            dist += path_Cost(tour)
            cus_temp, visit_temp = cus_Cost(tour)
            cus_time += cus_temp
        bus_dis_cost = int(dist * 3.17)
        cus_inveh_cost = int(cus_time * 0.87)
        bus_num_cost = int(bus_num * 0.37 * 120)
        cus_pena_cost = int(pena_num * cuspenalty)
        total = int(bus_num_cost + bus_dis_cost + cus_inveh_cost + cus_pena_cost)
        return total
    
    @property
    def cost(self):
        dist = 0
        cus_time = 0
        bus_num = busnum_check(self.routes)
        pena_num = len(self.penalty)
        for route in self.routes:
            tour = [0] + route + [0]
            dist += path_Cost(tour)
            cus_temp, visit_temp = cus_Cost(tour)
            cus_time += cus_temp
        bus_dis_cost = int(dist * 3.17)
        cus_inveh_cost = int(cus_time * 0.87)
        bus_num_cost = int(bus_num * 0.37 * 120)
        cus_pena_cost = int(pena_num * cuspenalty)
        return [bus_num_cost, bus_dis_cost, cus_inveh_cost, cus_pena_cost, bus_num]

    def objective_emis(self):
        route_emis = 0
        F = 0.0155
        f = 850

        self.routes = [route for route in self.routes if len(route) != 0]

        for ind, route in enumerate(self.routes):
            route = [0] + route + [0]
            for node in range(len(route) - 1):
                route_emis += (F * f) * self.distance[route[node]][route[node + 1]] / 4

        return int(route_emis)

    def objective_risk(self):
        route_risk = 0

        self.routes = [route for route in self.routes if len(route) != 0]

        for ind, route in enumerate(self.routes):
            route = [0] + route + [0]
            route_risk += sum(self.risk[route[idx]][route[idx+1]] for idx in range(len(route) - 1))
        return int(route_risk)

    def find_route(self, customer):
        requests = copy.deepcopy(req)
        for route in self.routes:
            if (requests[customer][0] in route) and (requests[customer][1] in route):
                return route

        raise ValueError(f"Solution does not contain customer {customer}.")
    
    def replaceRoute(self, index, route):
        self.routes[index] = route

def feasibility_check(route):
    feasibility = True
    check_htw, visit_time = tw_check(route)
    if check_htw > 0:
        feasibility = False
        return feasibility
    else:
        check_stw = re_check_ln(route, visit_time, reliable_a=0.9)
        if check_stw > 0:
            feasibility = False
            return feasibility
        else:
            check_dwell = dwell_check(route, visit_time, dwellmax=10)
            if check_dwell > 0:
                feasibility = False
                return feasibility
            else:
                check_load = load_check(route)
                if check_load > 0:
                    feasibility = False
                    return feasibility
                else:
                    return feasibility

def shuffle_routes(state):
    random.shuffle(state.routes)

def load_check(route):
    route_Load = 0
    for node in route:
        route_Load += dem[node]
        if route_Load - capacity > 0:
            return True
    return False

def tw_check(route):
    visit_time = []
    for ind, node in enumerate(route):
        if ind == 0:
            node_visit = max(0, tw[route[1]][0] - dis[node][route[1]])
            visit_time.append(node_visit)
        elif node in pick_list.keys():
            node_visit += dis[route[ind-1]][node] + sd[route[ind-1]]
            node_visit = max(node_visit, tw[node][0])
            visit_time.append(node_visit)
        else:
            node_visit += dis[route[ind-1]][node] + sd[route[ind-1]]
            visit_time.append(node_visit)    
    for ind, node in enumerate(route):
        if node in deli_list.keys():
            if visit_time[ind] - tw[node][1] > 0:
                return True, visit_time
    return False, visit_time

def dwell_check(route, visit_time, dwellmax):
    for ind, node in enumerate(route):
        if ind >0:
            node_dwell = visit_time[ind] - visit_time[ind-1] - dis[route[ind-1]][node]
            if node_dwell > dwellmax:
                return True
    return False

def re_check_ln(route, visit_time, reliable_a):
    for ind, node in enumerate(route):
        if ind == 0:
            node_var = 0
            node_mean = 0
            node_shift = 0
        elif ind <= 1:
            node_var += var[route[ind-1]][node]
            node_mean +=  dis[route[ind-1]][node]
            node_shift += visit_time[ind] - visit_time[ind-1] - dis[route[ind-1]][node]
        else:
            corr_temp = corr[locs[route[ind-2]], locs[route[ind-1]], locs[node]]
            node_var += var[route[ind-1]][node] + std[route[ind-1]][node] * std[route[ind-2]][route[ind-1]] * corr_temp
            node_mean +=  dis[route[ind-1]][node]
            node_shift += visit_time[ind] - visit_time[ind-1] - dis[route[ind-1]][node]
            if node in deli_list.keys():
                deli_mu_ln = math.log(node_mean ** 2 / math.sqrt(node_var + node_mean ** 2))
                deli_sigma_ln = math.sqrt(math.log(node_var/ node_mean ** 2 + 1))
                deli_reli = lognorm.ppf(q = reliable_a, s = deli_sigma_ln, \
                    loc = node_shift, scale = math.exp(deli_mu_ln))
                if (deli_reli - (tw[node][1] - visit_time[0])) > 0:
                    return True
    return False

def seq_check(route):
    check_seq = 0
    for ind, node in enumerate(route):
        if node in pick_list.keys():
            deli_node = pick_list[node]
            deli_index = route.index(deli_node)
            check_seq += max(ind - deli_index, 0)
    return check_seq

def busnum_check(routes_temp):
    routes_temp = [route for route in routes_temp if len(route) != 0]
    departure = []
    arrival = []
    trip_num = len(routes_temp)
    for route in routes_temp:
        route = [0] + route + [0]
        visit_time = visit_Calc(route)
        depart = visit_time[0]
        arrive = visit_time[-1]
        departure.append(depart)
        arrival.append(arrive)
    trip = 0
    for i in sorted(arrival):
        for j in sorted(departure):
            if j > i + 10:
                departure.remove(j)
                trip += 1
                break
    bus_num = trip_num - trip
    return bus_num

def visit_Calc(route):
    visit_time = []
    for ind, node in enumerate(route):
        if ind == 0:
            node_visit = max(0, tw[route[1]][0] - dis[node][route[1]])
            visit_time.append(node_visit)
        elif node in pick_list.keys():
            node_visit += dis[route[ind-1]][node] + sd[route[ind-1]]
            node_visit = max(node_visit, tw[node][0])
            visit_time.append(node_visit)
        else:
            node_visit += dis[route[ind-1]][node] + sd[route[ind-1]]
            visit_time.append(node_visit)
    return visit_time

# ======== random removal =========
def random_removal(state, rnd_state):
    if len(req) < 100:
        destroy_rate = 0.2
    else:
        destroy_rate = 0.1
    destroyed = copy.deepcopy(state)
    total_req = list(req.keys())
    penalty_req = copy.deepcopy(destroyed.penalty)
    assigned_req = [x for x in total_req if x not in penalty_req]
    customers_to_remove = max(int((len(total_req)) * destroy_rate) - len(penalty_req), 0) + 1
    
    for customer in rnd_state.choice(
        assigned_req, customers_to_remove, replace=False
    ):
        requests = copy.deepcopy(req)
        destroyed.unassigned.append(customer)
        route = destroyed.find_route(customer)
        route.remove(requests[customer][0])
        route.remove(requests[customer][1])

    return destroyed

# ======== worst removal =========
def worst_removal(state, rnd_state):

    requests = copy.deepcopy(req)
    if len(requests) < 100:
        destroy_rate = 0.2
    else:
        destroy_rate = 0.1
    destroyed = copy.deepcopy(state)

    total_req = list(requests.keys())
    penalty_req = copy.deepcopy(destroyed.penalty)
    assigned_req = [x for x in total_req if x not in penalty_req]

    current_routes =  copy.deepcopy(state.routes)
    
    worst_customers = sorted(assigned_req,
                         key=lambda temp: destroy_cost(temp, current_routes), reverse = True)
    customers_to_remove = max(int((len(total_req)) * destroy_rate) - len(penalty_req), 0) + 1
    for idx in range(customers_to_remove):
        customer_temp = worst_customers[idx]
        destroyed.unassigned.append(customer_temp)
        route = destroyed.find_route(customer_temp)
        route.remove(requests[customer_temp][0])
        route.remove(requests[customer_temp][1])
    return destroyed

def destroy_cost(customer, current_routes): # todo
    requests = copy.deepcopy(req)
    route_temp = find_current_route(customer, current_routes)
    route_temp.insert(0, 0)
    route_temp.append(0)
    pick_temp = route_temp.index(requests[customer][0])
    deli_temp = route_temp.index(requests[customer][1])
    new_cost = dis[route_temp[pick_temp-1]][route_temp[pick_temp+1]] + dis[route_temp[deli_temp-1]][route_temp[deli_temp+1]]
    old_cost = dis[route_temp[pick_temp-1]][route_temp[pick_temp]] + dis[route_temp[pick_temp]][route_temp[pick_temp+1]] + \
        dis[route_temp[deli_temp-1]][route_temp[deli_temp]] + dis[route_temp[deli_temp]][route_temp[deli_temp+1]]
    return old_cost - new_cost

# ======== substring removal =========
def string_removal(state, rnd_state):
    requests = copy.deepcopy(req)
    if len(requests) < 100:
        destroy_rate = 0.2
    else:
        destroy_rate = 0.1

    destroyed = state.copy()
    
    avg_route_size = int(np.mean([len(route) for route in state.routes]))

    destroyed_routes = []
    total_req = list(requests.keys())
    penalty_req = copy.deepcopy(destroyed.penalty)
    assigned_req = [x for x in total_req if x not in penalty_req]
    customers_to_remove = max(int((len(total_req)) * destroy_rate) - len(penalty_req), 0) + 1
    center = random.choice(assigned_req)
    adja_cen = [i for i in adjacent[center] if i in assigned_req]
    len_cus = 0
    for customer in adja_cen:
        if len_cus >= customers_to_remove:
            break
        if customer in destroyed.unassigned:
            continue
        route = destroyed.find_route(customer)
        if route in destroyed_routes:
            continue
        customers = remove_string(route, customer, avg_route_size, rnd_state)
        len_cus += len(customers)
        destroyed.unassigned.extend(customers)
        destroyed_routes.append(route)
    return destroyed

def remove_string(route, cust, max_string_size, rnd_state):
    requests = copy.deepcopy(req)

    size = rnd_state.randint(1, min(len(route), max_string_size) + 1)
    adja_cust = [i for i in adjacent[cust] if i in route][:size]
    adja_cust.insert(0, cust) 
    for customer in adja_cust:
        route.remove(requests[customer][0])
        route.remove(requests[customer][1])
    
    return adja_cust

def can_insert(customer, route, pick_idx, deli_idx):
    requests = copy.deepcopy(req)
    route_temp = copy.deepcopy(route)
    route_temp.insert(pick_idx, requests[customer][0])
    route_temp.insert(deli_idx, requests[customer][1])
    route_temp.insert(0, 0)
    route_temp.append(0)
    return feasibility_check(route_temp)

def insert_cost(customer, route, pick_idx, deli_idx):
    requests = copy.deepcopy(req)
    old_cost = route_cost(route)
    new_route = copy.deepcopy(route)
    new_route.insert(pick_idx, requests[customer][0])
    new_route.insert(deli_idx, requests[customer][1])
    new_cost = route_cost(new_route)
    cost = new_cost - old_cost
    return cost

# ======== Random repair =========
def random_repair(state, rnd_state):
    state.unassigned = state.unassigned + state.penalty
    state.penalty = []
    rnd_state.shuffle(state.unassigned)
    requests = copy.deepcopy(req)
    while len(state.unassigned) != 0:
        customer = state.unassigned.pop()
        first = float('inf')
        best_ind, best_route, best_pick, best_deli = None, None, None, None
        for ind, route in enumerate(state.routes):
            if len(route) <= math.ceil(np.mean([len(temp) for temp in state.routes])):
                visit_time = visit_route(route)
                visit_time = visit_time[1:-1]
                for pick_idx in range(len(route) + 1):
                    for deli_idx in range(pick_idx + 1, len(route) + 2):
                        if (pick_idx <len(route)) and (deli_idx <len(route)):
                            if (visit_time[pick_idx] >= tw[requests[customer][0]][0]) and (visit_time[deli_idx] <= tw[requests[customer][1]][1]):
                                if can_insert(customer, route, pick_idx, deli_idx):
                                    cost = insert_cost(customer, route, pick_idx, deli_idx)
                                    if cost < first:
                                        best_ind, best_route, best_pick, best_deli = ind, route, pick_idx, deli_idx
                                        first = cost
                        else:
                            if can_insert(customer, route, pick_idx, deli_idx):
                                cost = insert_cost(customer, route, pick_idx, deli_idx)
                                if cost < first:
                                    best_ind, best_route, best_pick, best_deli = ind, route, pick_idx, deli_idx
                                    first = cost
        if first < float("inf"):
            temproute = copy.deepcopy(best_route)
            temproute.insert(best_pick, requests[customer][0])
            temproute.insert(best_deli, requests[customer][1])
            state.replaceRoute(best_ind, temproute)
        else:
            routes_temp = copy.deepcopy(state.routes)
            route_temp = [0, requests[customer][0], requests[customer][1], 0]
            routes_temp.append(requests[customer])
            bus_number = busnum_check(routes_temp)
            if bus_number <= len(veh) and feasibility_check(route_temp) != 0:
                state.routes.append(requests[customer])
            else:
                state.penalty.append(customer)

    return remove_empty_routes(state)

# ======== Greedy repair =========
def greedy_repair(state, rnd_state):
    state.unassigned = state.unassigned + state.penalty
    state.penalty = []
    requests = copy.deepcopy(req)
    n_insert = []
    bestPoses = {}
    for customer in state.unassigned:
        bestPoses[customer] = []
        first = float('inf')
        best_ind, best_route, best_pick, best_deli = None, None, None, None
        for ind, route in enumerate(state.routes):
            if len(route) <= math.ceil(np.mean([len(temp) for temp in state.routes])):
                visit_time = visit_route(route)
                visit_time = visit_time[1:-1]
                for pick_idx in range(len(route) + 1):
                    for deli_idx in range(pick_idx + 1, len(route) + 2):
                        if (pick_idx <len(route)) and (deli_idx <len(route)):
                            if (visit_time[pick_idx] >= tw[requests[customer][0]][0]) and (visit_time[deli_idx] <= tw[requests[customer][1]][1]):
                                if can_insert(customer, route, pick_idx, deli_idx):
                                    cost = insert_cost(customer, route, pick_idx, deli_idx)
                                    bestPoses[customer].append([cost, ind, route, pick_idx, deli_idx])
                                    if cost < first:
                                        best_ind, best_route, best_pick, best_deli = ind, route, pick_idx, deli_idx
                                        first = cost
                        else:
                            if can_insert(customer, route, pick_idx, deli_idx):
                                cost = insert_cost(customer, route, pick_idx, deli_idx)
                                bestPoses[customer].append([cost, ind, route, pick_idx, deli_idx])
                                if cost < first:
                                    best_ind, best_route, best_pick, best_deli = ind, route, pick_idx, deli_idx
                                    first = cost

       n_insert.append([customer, first, best_ind, best_route, best_pick, best_deli])
    bestPoses = {k:v for k,v in bestPoses.items() if v}
    n_insert.sort(key=lambda x: x[1])

    for i in n_insert:
        state.unassigned.remove(i[0])
        if i[3] is not None:
            temproute = copy.deepcopy(state.routes[i[2]])
            if can_insert(i[0], temproute, i[4], i[5]):
                temproute.insert(i[4], requests[i[0]][0])
                temproute.insert(i[5], requests[i[0]][1])
                state.replaceRoute(i[2], temproute)
            else:
                short_index, short_route, short_pick, short_deli = shortest_insert(i[0], state, bestPoses)
                if short_pick is None:
                    routes_temp = copy.deepcopy(state.routes)
                    route_temp = [0, requests[i[0]][0], requests[i[0]][1], 0]
                    routes_temp.append(requests[i[0]])
                    bus_number = busnum_check(routes_temp)
                    if bus_number <= len(veh) and feasibility_check(route_temp) != 0:
                        state.routes.append(requests[i[0]])
                    else:
                        state.penalty.append(i[0])
                else:
                    short_route.insert(short_pick, requests[i[0]][0])
                    short_route.insert(short_deli, requests[i[0]][1])
                    state.replaceRoute(short_index, short_route)        
        else:
            routes_temp = copy.deepcopy(state.routes)
            route_temp = [0, requests[i[0]][0], requests[i[0]][1], 0]
            routes_temp.append(requests[i[0]])
            bus_number = busnum_check(routes_temp)
            if bus_number <= len(veh) and feasibility_check(route_temp) != 0:
                state.routes.append(requests[i[0]])
            else:
                state.penalty.append(i[0])

    return remove_empty_routes(state)

# ======== regret repair =========
def regret_repair(state, rnd_state):
    state.unassigned = state.unassigned + state.penalty
    state.penalty = []
    requests = copy.deepcopy(req)
    n_insert = []
    bestPoses = {}
    for customer in state.unassigned:
        bestPoses[customer] = []
        first = second = float('inf')
        best_ind, best_route, best_pick, best_deli = None, None, None, None
        for ind, route in enumerate(state.routes):
            if len(route) <= math.ceil(np.mean([len(temp) for temp in state.routes])):
                visit_time = visit_route(route)
                visit_time = visit_time[1:-1]
                for pick_idx in range(len(route) + 1):
                    for deli_idx in range(pick_idx + 1, len(route) + 2):
                        if (pick_idx <len(route)) and (deli_idx <len(route)):
                            if (visit_time[pick_idx] >= tw[requests[customer][0]][0]) and (visit_time[deli_idx] <= tw[requests[customer][1]][1]):
                                if can_insert(customer, route, pick_idx, deli_idx):
                                    cost = insert_cost(customer, route, pick_idx, deli_idx)
                                    bestPoses[customer].append([cost, ind, route, pick_idx, deli_idx])
                                    if cost < first:
                                        best_ind, best_route, best_pick, best_deli = ind, route, pick_idx, deli_idx
                                        first = cost 
                                        second = first
                                    elif(cost  < second and cost != first):
                                        second = cost
                        else:
                            if can_insert(customer, route, pick_idx, deli_idx):
                                cost = insert_cost(customer, route, pick_idx, deli_idx)
                                bestPoses[customer].append([cost, ind, route, pick_idx, deli_idx])
                                if cost < first:
                                    best_ind, best_route, best_pick, best_deli = ind, route, pick_idx, deli_idx
                                    first = cost 
                                    second = first
                                elif(cost  < second and cost != first):
                                    second = cost

        val = float('%.2f' % (second - first))
        n_insert.append([customer, val, best_ind, best_route, best_pick, best_deli])
    bestPoses = {k:v for k,v in bestPoses.items() if v}
    n_insert.sort(key=lambda x: x[1], reverse=True)

    for i in n_insert:
        state.unassigned.remove(i[0])
        if i[3] is not None:
            temproute = copy.deepcopy(state.routes[i[2]])

            if can_insert(i[0], temproute, i[4], i[5]):
                temproute.insert(i[4], requests[i[0]][0])
                temproute.insert(i[5], requests[i[0]][1])
                state.replaceRoute(i[2], temproute)
            else:
                short_index, short_route, short_pick, short_deli = shortest_insert(i[0], state, bestPoses)
                if short_pick is None:
                    routes_temp = copy.deepcopy(state.routes)
                    route_temp = [0, requests[i[0]][0], requests[i[0]][1], 0]
                    routes_temp.append(requests[i[0]])
                    bus_number = busnum_check(routes_temp)
                    if bus_number <= len(veh) and feasibility_check(route_temp) != 0:
                        state.routes.append(requests[i[0]])
                    else:
                        state.penalty.append(i[0])
                else:
                    short_route.insert(short_pick, requests[i[0]][0])
                    short_route.insert(short_deli, requests[i[0]][1])
                    state.replaceRoute(short_index, short_route)        
        else:
            routes_temp = copy.deepcopy(state.routes)
            route_temp = [0, requests[i[0]][0], requests[i[0]][1], 0]
            routes_temp.append(requests[i[0]])
            bus_number = busnum_check(routes_temp)
            if bus_number <= len(veh) and feasibility_check(route_temp) != 0:
                state.routes.append(requests[i[0]])
            else:
                state.penalty.append(i[0])

    return remove_empty_routes(state)

def shortest_insert(customer, state, bestPoses):
    routes_temp = copy.deepcopy(state.routes)
    short_index, short_route, short_pick, short_deli = None, None, None, None
    bestPoses[customer].sort(key=lambda x: x[0])
    temp = []
    breakout = False
    for i in bestPoses[customer]:
        if i[1] in temp:
            continue
        else:
            temp.append(i[1])
            for pick_idx in range(len(routes_temp[i[1]]) + 1):
                for deli_idx in range(pick_idx + 1, len(routes_temp[i[1]]) + 2):
                    if can_insert(customer, routes_temp[i[1]], pick_idx, deli_idx):
                        short_index, short_route, short_pick, short_deli = i[1], routes_temp[i[1]], pick_idx, deli_idx
                        breakout = True
                        break
                if breakout:
                    break
            if breakout:
                break
    return short_index, short_route, short_pick, short_deli

# ======== initial solution =========
def neighbors(node):
    locations = np.argsort(dis[node])
    return locations[locations != 0]

def nearest_neighbor_1by1():
    routes = []
    requests = copy.deepcopy(req)
    unvisited = [requests[customer][0] for customer in requests.keys()]

    veh_index = 0
    while unvisited:
        route = [0]
        veh_index += 1

        while unvisited:
            current = route[-1]
            nearest = [nb for nb in neighbors(current) if nb in unvisited][0]
            route_temp = copy.deepcopy(route)
            route_temp.append(nearest)
            route_temp.append(pick_list[nearest])
            route_temp.append(0)
            if not feasibility_check(route_temp):
                break
            route.append(nearest)
            route.append(pick_list[nearest])
            unvisited.remove(nearest)
        customers = route[1:]
        routes.append(customers)
    
    return CpdptwState(routes)

def greedy_insert(customer, routes):
    best_cost, best_route, best_pick, best_deli = None, None, None, None
    for route in routes:
        for pick_idx in range(len(route) + 1):
            for deli_idx in range(pick_idx + 1, len(route) + 2):
                if can_insert(customer, route, pick_idx, deli_idx):
                    cost = insert_cost(customer, route, pick_idx, deli_idx)
                    if best_cost is None or cost < best_cost:
                        best_cost, best_route, best_pick, best_deli = cost, route, pick_idx, deli_idx

    return best_route, best_pick, best_deli

def re_Calc_ln(route, visit_time):
    prob = []
    reli_time = []
    for ind, node in enumerate(route):
        if ind == 0:
            node_var = 0
            node_mean = 0
            node_shift = 0
        elif ind <= 1:
            node_var += var[route[ind-1]][node]
            node_mean +=  dis[route[ind-1]][node]
            node_shift += visit_time[ind] - visit_time[ind-1] - dis[route[ind-1]][node]
        else:
            corr_temp = corr[locs[route[ind-2]], locs[route[ind-1]], locs[node]]
            node_var += var[route[ind-1]][node] + std[route[ind-1]][node] * std[route[ind-2]][route[ind-1]] * corr_temp
            node_mean +=  dis[route[ind-1]][node]
            node_shift += visit_time[ind] - visit_time[ind-1] - dis[route[ind-1]][node]
            if node in deli_list.keys():
                deli_mu_ln = math.log(node_mean ** 2 / math.sqrt(node_var + node_mean ** 2))
                deli_sigma_ln = math.sqrt(math.log(node_var/ node_mean ** 2 + 1))
                temp = (tw[node][1] - visit_time[0])
                prob_temp = lognorm.cdf(x = temp, s = deli_sigma_ln, loc = node_shift, scale = math.exp(deli_mu_ln))
                prob.append(prob_temp)
                if prob_temp >= 0.9:
                    reli_time.append(0)
                else:
                    temp1 = lognorm.ppf(q = 0.9, s = deli_sigma_ln, loc = node_shift, scale = math.exp(deli_mu_ln))
                    reli_time.append(round(temp1 - temp, 2))
    return prob, reli_time

if __name__ == '__main__':
    print('start')

    global req, dis, veh, dem, tw, sd, var, std, mu_ln, sigma_ln, pick_list, deli_list, \
        adjacent, corr, locs, capacity, cuspenalty, risk
    cuspenalty = 500

    calculation_time = [1200, 1200, 1200, 1200]
    max_iterations = [1000, 1000, 500, 500]
    size = [51, 101, 201, 401]
    node = [23, 23, 23, 23]

    all_coding_time = []

    for z in range(num_z):
        coding_time = []

        for index in range(4):
            inst = [node[index], 8, 15, size[index], 120, 1, 1, 10, 15]
            tc_duration = inst[4]
            total_requests = inst[3] - 1
            capacity = inst[7]
            speed = inst[8]
            asb_num = int(math.ceil(
                round((total_requests) / capacity)) * 1.5)
            node_num = inst[3]
            net_num = inst[0]
            calc_time = calculation_time[index]

            file_path = os.path.join('datatest', 'newmatrix%s_%s.txt' \
                                     % (total_requests, tc_duration))
            data_path = os.path.abspath(file_path)
            veh, task_no_list, locs, dem, tw, sd, req, dis, var, std, mu_ln, \
                sigma_ln, pick_list, deli_list, corr = (read_pdptw_benchmark_data \
                                                            (data_path, asb_num, node_num, net_num, capacity, speed))

            file_risk_path = os.path.join('datatest', 'risk_matrix_%s_1.csv' \
                                          % (total_requests))
            data_risk_path = os.path.abspath(file_risk_path)
            risk = read_risk(data_risk_path)

            adjacent = {}
            for i in list(req.keys()):
                adjacent[i] = adjacent_customers(list(req.keys()), i)
            # ======== ALNS算法构建 =========
            SEED = 1234

            start = time.time()
            alns = ALNS(rnd.RandomState(SEED))
            alns.add_destroy_operator(random_removal)
            alns.add_destroy_operator(worst_removal)
            alns.add_destroy_operator(string_removal)
            alns.add_repair_operator(random_repair)
            alns.add_repair_operator(greedy_repair)
            alns.add_repair_operator(regret_repair)
            init = nearest_neighbor(dem, dis, risk)
            select = SegmentedRouletteWheel(scores=[25, 5, 1, 0],
                                            decay=0.8,
                                            seg_length=500,
                                            num_destroy=3,
                                            num_repair=3)
            accept = SimulatedAnnealing(start_temperature=1_000,
                                        end_temperature=1,
                                        step=0.999,
                                        method="exponential")
            if index <= 2:
                stop = MaxIterations(num_interations)
            else:
                stop = MaxRuntime(max_time)

            result, Pareto_Best_Objective_All, Pareto_Accept_Objective_All, Pareto_Best_Container = \
                alns.iterate(init, select, accept, stop)

            end = time.time()

