import numpy as np
import sys
import os
from copy import copy
from copy import deepcopy
from utils import pretty_print_timetable, read_yaml_file
from utils import ZILE, SALI, INTERVALE, MATERII, PROFESORI
from heapq import heapify, heappop
from check_constraints import check_optional_constraints, check_mandatory_constraints
from time import time
from math import sqrt, log

DISCIPLINES = 'Disciplines'
CONSTRAINTS = 'Constraints'
DAYS = 'Days'
INTERVALS = 'Intervals'
PREFFERED = "Preffered"
UNPREFFERED = "Unpreffered"
PAUSE = "Pause"
N = 'N'
Q = 'Q'
PARENT = 'Parent'
STATE = 'State'
ACTIONS = 'Actions'

def get_intervals(interval):
    s = interval.split('-')
    if "!" in s[0]:
        start = int(s[0][1:])
        stop = int(s[1])
        return [f"{x}-{x + 2}" for x in range(start, stop, 2)]
    else:
        start = int(s[0])
        stop = int(s[1])
        return [f"{x}-{x + 2}" for x in range(start, stop, 2)]

class State:
    def __init__(self, timetable, nconflicts, teachers_hours, attended_hours, teachers_intervals):
        self.timetable = timetable
        self.nconflicts = nconflicts
        self.teachers_hours = teachers_hours
        self.attended_hours = attended_hours
        self.teachers_intervals = teachers_intervals

    def apply_action(self, day, interval, discipline):
        formatted_interval = str(interval)[1:(len(str(interval)) - 1)]
        formatted_interval = formatted_interval.replace(", ", "-")
        possible_results = []
        for classroom in classrooms:
            if self.timetable[day][interval][classroom] != () or discipline not in classrooms[classroom][MATERII]: continue
            for teacher in teachers:
                if f"!{day}" in constraints[teacher][CONSTRAINTS][DAYS]: continue
                state = deepcopy(self)

                if discipline not in teachers[teacher][MATERII]: continue
                if len([x[0] for x in state.timetable[day][interval].values() if x != () and x[0] == teacher]) == 1: continue
                if state.teachers_hours[teacher] == 7: continue
                if f"{formatted_interval}" in constraints[teacher][CONSTRAINTS][UNPREFFERED]: state.nconflicts += 1
                if constraints[teacher][CONSTRAINTS][PAUSE]:
                    hours = int(constraints[teacher][CONSTRAINTS][PAUSE][0].split(' ')[2])
                    length = len(state.teachers_intervals[teacher][day])
                    if length > 0 and interval[0] - state.teachers_intervals[teacher][day][length - 1][1] > hours:
                        state.nconflicts += hours

                state.timetable[day][interval][classroom] = (teacher, discipline)
                state.teachers_hours[teacher] += 1
                state.attended_hours[discipline] -= classrooms[classroom]['Capacitate']
                state.teachers_intervals[teacher][day].append(interval)

                state = State(state.timetable, state.nconflicts, state.teachers_hours, state.attended_hours, state.teachers_intervals)
                possible_results.append(state)

        return possible_results

    def f(self, discipline):
        return self.attended_hours[discipline] + 14 * self.conflicts()

    def get_next_states(self, discipline):
        next_states = []

        for day in days:
            for interval in intervals:
                if self.attended_hours[discipline] > 0:
                    next_states += self.apply_action(day, eval(interval), discipline)

        return next_states

    def conflicts(self):
        return self.nconflicts

    def is_final(self):
        return self.nconflicts == 0 and all(self.is_final_discipline(d) for d in disciplines)

    def is_final_discipline(self, discipline):
        return self.attended_hours[discipline] <= 0

    def clone(self):
        return State(copy(self.timetable), self.nconflicts, self.teachers_hours, self.attended_hours, self.teachers_intervals)

################################# HILL CLIMBING ########################################

def count_disciplines_classrooms(discipline):
    return len([classroom for classroom in classrooms if discipline in classrooms[classroom][MATERII]])

def hill_climbing(initial: State, max_iters: int = 1000):
    iters, states = 0, 0

    state = copy(initial)
    subjects = [(count_disciplines_classrooms(discipline), discipline) for discipline in disciplines]
    heapify(subjects)

    while subjects:
        _, discipline = heappop(subjects)
        while iters < max_iters:
            iters += 1

            next_states = state.get_next_states(discipline)

            next_states_conflicts = [(s, s.f(discipline)) for s in next_states]
            states += len(next_states_conflicts)

            if next_states_conflicts == []:
                break

            min_state_conflict = next_states_conflicts[0][1]
            aux_state = next_states_conflicts[0][0]

            for (s, c) in next_states_conflicts:
                if c < min_state_conflict:
                    min_state_conflict = c
                    aux_state = s

            if min_state_conflict >= state.f(discipline):
                break
            else:
                state = aux_state

    return state.is_final(), iters, states, state



################################# MONTE CARLO ########################################

def init_node(state, parent = None):
    return {N: 0, Q: 0, STATE: state, PARENT: parent, ACTIONS: {}}

CP = 1.0 / sqrt(2.0)

def select_action(node, c = CP):
    N_node = node[N]

    max_action = None
    max_ucb = -float('inf')

    for action, child in node[ACTIONS].items():
        if child[N] == 0:
            current_ucb = float('inf')
        else:
            current_ucb = child[Q] / child[N] + c * sqrt(2 * log(N_node) / child[N])
        if not max_action or current_ucb > max_ucb:
            max_action = action
            max_ucb = current_ucb

    return max_action

def softmax(x: np.array) -> float:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def mcts(state0, budget, tree, discipline):
    global block
    global iters
    global states_number
    tree = tree if tree else init_node(state0)
    states = state0.get_next_states(discipline)
    arr = np.array([-state.f(discipline) for state in states if state])
    states_number += len(states)

    for _ in range(budget):
        node = tree
        s = state0

        while s and not s.is_final_discipline(discipline) and all(action in node[ACTIONS] for action in states):
            new_action = select_action(node)
            if new_action:
                s = new_action
                node = node[ACTIONS][new_action]
            else:
                break

        if s and not s.is_final_discipline(discipline):
            if states:
                idx = np.random.choice(len(states), p=softmax(arr))
                s = states[idx]
                node = init_node(s, node)
                node[PARENT][ACTIONS][s] = node
            else:
                break

        s = node[STATE]
        if s:
            while s and not s.is_final_discipline(discipline):
                next_states = s.get_next_states(discipline)
                states_number += len(next_states)
                if next_states:
                    arr1 = np.array([-state.f(discipline) for state in next_states if state])
                    idx = np.random.choice(len(next_states), p=softmax(arr1))
                    s = next_states[idx]
                    if s not in node[ACTIONS]:
                        node[ACTIONS][s] = init_node(s, node)
                    node = node[ACTIONS][s]
                else:
                    break

        if s:
            if s.is_final():
                reward = -s.f(discipline)
            if s.is_final_discipline(discipline):
                reward = -s.f(discipline) / 10
            else:
                reward = s.f(discipline)
            aux = node
            while aux:
                aux[N] += 1
                aux[Q] += reward
                aux = aux[PARENT]

            if s.is_final():
                break

        if node[N] < 10:
            break

    if tree:
        final_action = select_action(tree, 0.0)
        if final_action:
            return (final_action, tree[ACTIONS][final_action])
        else:
            block = True

    if state0.get_next_states(discipline):
        return (state0.get_next_states(discipline)[0], init_node())
    return (state0, init_node(state0))

if __name__ == '__main__':
    used_algorithm = sys.argv[1]
    input_file = sys.argv[2]

    yaml_dict = read_yaml_file(input_file)

    days = yaml_dict[ZILE]
    intervals = yaml_dict[INTERVALE]
    classrooms = yaml_dict[SALI]
    disciplines = yaml_dict[MATERII]
    teachers = yaml_dict[PROFESORI]

    no_days = list(map(lambda x: '!' + x, days))

    constraints = {
        teacher: {
            DISCIPLINES: teachers[teacher][MATERII],
            CONSTRAINTS: {
                DAYS: [x for x in teachers[teacher]['Constrangeri'] if x in days or x in no_days],
                PREFFERED: [interval for x in teachers[teacher]['Constrangeri'] 
                            if "-" in x and "!" not in x 
                            for interval in get_intervals(x)],
                UNPREFFERED: [interval for x in teachers[teacher]['Constrangeri'] 
                            if "!" in x and "-" in x 
                            for interval in get_intervals(x)],
                PAUSE: [p for p in teachers[teacher]['Constrangeri'] if "Pauza" in p]
            }
        }
        for teacher in teachers
    }

    state = State({day: {eval(interval): {classroom: () for classroom in classrooms} for interval in intervals} for day in days},
                0,
                {t: 0 for t in teachers},
                {d: n for d, n in disciplines.items()},
                {t: {d: [] for d in days} for t in teachers})

    if not os.path.exists(f"./outputs/{used_algorithm}"):
        os.makedirs("./outputs/hc")

    filename = input_file.split('/')[-1]
    output_file = filename.split('.')[0]

    materii = [discipline for discipline in disciplines]
    subjects = sorted(materii, key=lambda x: (count_disciplines_classrooms(x), disciplines[x], len([t for t in teachers if x in teachers[t][MATERII]])))

    if used_algorithm == "hc":
        start_time = time()
        _, iters, states, state = hill_climbing(state)
        end_time = time()
        with open(f'./outputs/hc/{output_file}.txt', 'w') as f:
            f.write(pretty_print_timetable(state.timetable, input_file))
            f.write(f"Total number of iterations to reach the solution: {iters}\n")
            f.write(f"Total number of states generated to reach the solution: {states}\n")
            f.write(f"Total number of soft conflicts for final solution: {state.conflicts()}\n")
            f.write(f"Checker hard constraints: {check_mandatory_constraints(state.timetable, yaml_dict)}\n")
            f.write(f"Checker soft constraints: {check_optional_constraints(state.timetable, yaml_dict)}\n")
            f.write(f"Execution time: {end_time - start_time} seconds.\n")
    elif used_algorithm == "mcts":
        start_time = time()
        tree = None
        block = False
        iters = 0
        states_number = 0
        for discipline in subjects:
            while not state.is_final_discipline(discipline):
                iters += 1
                state, tree = mcts(state, 20, tree, discipline)
                if block:
                    break
        end_time = time()
        with open(f'./outputs/mcts/{output_file}.txt', 'w') as f:
            f.write(pretty_print_timetable(state.timetable, input_file))
            f.write(f"Total number of iterations to reach the solution: {iters}\n")
            f.write(f"Total number of states generated to reach the solution: {states_number}\n")
            f.write(f"Total number of soft conflicts for final solution: {state.conflicts()}\n")
            f.write(f"Checker hard constraints: {check_mandatory_constraints(state.timetable, yaml_dict)}\n")
            f.write(f"Checker soft constraints: {check_optional_constraints(state.timetable, yaml_dict)}\n")
            f.write(f"Execution time: {end_time - start_time} seconds.\n")