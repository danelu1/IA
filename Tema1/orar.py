from utils import pretty_print_timetable, read_yaml_file
from utils import ZILE, SALI, INTERVALE, MATERII, PROFESORI
from copy import deepcopy
from heapq import heapify, heappop
from check_constraints import check_optional_constraints, check_mandatory_constraints
from time import time

DISCIPLINES = 'Disciplines'
CONSTRAINTS = 'Constraints'
DAYS = 'Days'
INTERVALS = 'Intervals'
PREFFERED = "Preffered"
UNPREFFERED = "Unpreffered"
PAUSE = "Pause"

start_time = time()

yaml_dict = read_yaml_file('inputs/orar_bonus_exact.yaml')

days = yaml_dict[ZILE]
intervals = yaml_dict[INTERVALE]
classrooms = yaml_dict[SALI]
disciplines = yaml_dict[MATERII]
teachers = yaml_dict[PROFESORI]

no_days = list(map(lambda x: '!' + x, days))

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
                state = deepcopy(self)

                if discipline not in teachers[teacher][MATERII]: continue
                if len([x[0] for x in state.timetable[day][interval].values() if x != () and x[0] == teacher]) == 1: continue
                if state.teachers_hours[teacher] == 7: continue
                if f"!{day}" in constraints[teacher][CONSTRAINTS][DAYS] and \
                    f"{formatted_interval}" in constraints[teacher][CONSTRAINTS][UNPREFFERED]: state.nconflicts += len(intervals) + 1
                elif f"!{day}" in teachers[teacher]['Constrangeri']: state.nconflicts += len(intervals)
                elif f"{formatted_interval}" in constraints[teacher][CONSTRAINTS][UNPREFFERED]: state.nconflicts += 1
                if constraints[teacher][CONSTRAINTS][PAUSE]:
                    pause = constraints[teacher][CONSTRAINTS][PAUSE][0]
                    hours = int(pause.split(' ')[2])
                    length = len(self.teachers_intervals[teacher][day])
                    if length > 0 and interval[0] - state.teachers_intervals[teacher][day][length - 1][1] < hours:
                        state.nconflicts += 1

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
        state = deepcopy(self)
        next_states = []

        for day in days:
            for interval in intervals:
                if self.attended_hours[discipline] > 0:
                    next_states += state.apply_action(day, eval(interval), discipline)

        return next_states

    def conflicts(self):
        return self.nconflicts

    def is_final(self):
        return self.nconflicts == 0 and all(self.is_final_discipline(d) for d in disciplines)

    def is_final_discipline(self, discipline):
        return self.attended_hours[discipline] <= 0

    def clone(self):
        return State(deepcopy(self.timetable), self.nconflicts, self.teachers_hours, self.attended_hours, self.teachers_intervals)

def count_disciplines_classrooms(discipline):
    return len([classroom for classroom in classrooms if discipline in classrooms[classroom][MATERII]])

def hill_climbing(initial: State, max_iters: int = 1000):
    iters, states = 0, 0

    state = initial.clone()
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

state = State({day: {eval(interval): {classroom: () for classroom in classrooms} for interval in intervals} for day in days},
              0,
              {t: 0 for t in teachers},
              {d: n for d, n in disciplines.items()},
              {t: {d: [] for d in days} for t in teachers})

_, iters, states, state = hill_climbing(state)
print(f"Total number of iterations to reach the solution: {iters}")
print(f"Total number of states generated to reach the solution: {states}")
print(f"Total number of soft conflicts for final solution: {state.conflicts()}")
print(pretty_print_timetable(state.timetable, './inputs/orar_bonus_exact.yaml'))
print(check_mandatory_constraints(state.timetable, yaml_dict))
print(check_optional_constraints(state.timetable, yaml_dict))

end_time = time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds.")