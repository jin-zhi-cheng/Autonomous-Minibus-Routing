from alns.accept.update import update
import math

# update pareto solution set
def Pareto_renew(rnd_state, cand, accept, Pareto_Best_Container, Pareto_Accept_Container, \
                 Pareto_Best_Objective_All, Pareto_Accept_Objective_All, num_iter):
    cand_set_objective = set_objective(cand)

    for i in range(len(Pareto_Best_Container)):
        if cand_set_objective == Pareto_Best_Container[i][0]:
            if compare_lists(cand.routes, Pareto_Best_Container[i][1]):
                for i in range(len(Pareto_Best_Container)):
                    Pareto_Best_Objective_All.append([Pareto_Best_Container[i][0], num_iter])

                if len(Pareto_Accept_Container):
                    for i in range(len(Pareto_Accept_Container)):
                        Pareto_Accept_Objective_All.append([Pareto_Accept_Container[i][0], num_iter])
                return Pareto_Best_Container, Pareto_Accept_Container, \
                    Pareto_Best_Objective_All, Pareto_Accept_Objective_All

    index_best_dominated = []
    index_cand_dominated = []
    for i in range(len(Pareto_Best_Container)):
        best_set_objective = Pareto_Best_Container[i][0]
        if dominated(best_set_objective, cand_set_objective):
            index_cand_dominated.append(i)

        if dominated(cand_set_objective, best_set_objective):
            index_best_dominated.append(i)

    if len(index_cand_dominated):
        for i in range(len(index_cand_dominated)):
            Pareto_Accept_Container.append(Pareto_Best_Container[index_cand_dominated[i]])
            Pareto_Best_Container.remove(Pareto_Best_Container[index_cand_dominated[i]])
            for j in range(len(index_cand_dominated)):
                index_cand_dominated[j] = index_cand_dominated[j] - 1

        Pareto_Best_Container.append([set_objective(cand), cand.routes, \
                                     cand.unassigned, cand.penalty])
    else:
        if len(index_best_dominated):
            Pareto_Accept_Container.append([set_objective(cand), cand.routes, \
                                            cand.unassigned, cand.penalty])

        else:
            Pareto_Best_Container.append([set_objective(cand), cand.routes, \
                                          cand.unassigned, cand.penalty])

    min_best = min_pareto_best(Pareto_Best_Container)

    accept._temperature = max(
        accept.end_temperature,
        update(accept._temperature, accept.step, accept.method),
    )

    temperature = accept._temperature
    remove = []
    for i in range(len(Pareto_Accept_Container)):
        if judge_accept(min_best, Pareto_Accept_Container[i][0], temperature, rnd_state):
            continue
        else:
            remove.append(i)

    if len(remove):
        for i in range(len(remove)):
            Pareto_Accept_Container.remove(Pareto_Accept_Container[remove[i]])
            for j in range(len(remove)):
                remove[j] = remove[j] - 1

    for i in range(len(Pareto_Best_Container)):
        Pareto_Best_Objective_All.append([Pareto_Best_Container[i][0], num_iter])

    if len(Pareto_Accept_Container):
        for i in range(len(Pareto_Accept_Container)):
            Pareto_Accept_Objective_All.append([Pareto_Accept_Container[i][0], num_iter])

    return Pareto_Best_Container, Pareto_Accept_Container, Pareto_Best_Objective_All, Pareto_Accept_Objective_All

# improved SA criteria
def judge_accept(min_best, Pareto_Accept_Container, temperature, rnd_state):
    judge = []
    for i in range(3):
        best_energy = min_best[i]
        curr_energy = Pareto_Accept_Container[i]

        energy_difference = curr_energy - best_energy

        random_num = rnd_state.rand()
        if energy_difference < 0 or random_num < \
                math.exp(-energy_difference / temperature):
            judge.append(1)
        else:
            judge.append(0)

    if judge[0] == 1:
        if judge[1] == 1:
            if judge[2] == 1:
                return True
    return False

# compared best and curr
def replace_best_curr(rnd_state, best, curr, Pareto_Best_Container, Pareto_Accept_Container, \
    index_dominated):

    random_number = rnd_state.rand()
    Pareto_best_num = int(random_number*len(Pareto_Best_Container))
    best = class_value_renew(best, Pareto_Best_Container[Pareto_best_num])

    if len(index_dominated[Pareto_best_num]):
        random_number = rnd_state.rand()
        index_dominated_num = int(random_number*len(index_dominated[Pareto_best_num]))
        Pareto_accept_num = index_dominated[Pareto_best_num][index_dominated_num]
        curr = class_value_renew(curr, Pareto_Accept_Container[Pareto_accept_num])
    else:
        curr = best

    return best, curr
