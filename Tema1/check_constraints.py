import yaml
import argparse
import sys
from utils import read_yaml_file, get_profs_initials, pretty_print_timetable


##################### MACROURI #####################
INTERVALE = 'Intervale'
ZILE = 'Zile'
MATERII = 'Materii'
PROFESORI = 'Profesori'
SALI = 'Sali'
CAPACITATE = 'Capacitate'
CONSTRANGERI = 'Constrangeri'


#################### FUNCTII AUXILIARE ####################
def parse_interval(interval : str):
    '''
    Se parsează un interval de forma "Ora1 - Ora2" în cele 2 componente.
    '''

    intervals = interval.split('-')
    return int(intervals[0].strip()), int(intervals[1].strip())


def parse_subject_room_prof(subject_room_prof : str, nick_to_prof : dict):
    '''
    Se parsează un string de forma "Materie : Sala - Profesor" în cele 3 componente.
    '''

    if 'goala' in subject_room_prof:
        room = subject_room_prof.split('-')[0].strip()

        return None, room, None


    subject = subject_room_prof.split(':')[0].strip()
    room = subject_room_prof.split('(')[1].split('-')[0].strip()
    
    prof = subject_room_prof.split('-')[1][:-1].strip()
    prof = nick_to_prof[prof]
    return subject, room, prof


def get_timetable(timetable_specs : dict, output_name : str, debug_flag : bool = False):
    '''
    Pe baza specificațiilor din fișierul de intrare, se reprezintă intern orarul din fișierul de ieșire.
    '''
    timetable = {day : {eval(interval) : {} for interval in timetable_specs[INTERVALE]} for day in timetable_specs[ZILE]}

    _, initials_to_prof = get_profs_initials(timetable_specs[PROFESORI])
    
    if debug_flag:
        print(initials_to_prof)
        print()

    interval = None

    with open(output_name, 'r') as file:
        for line in file:
            if line[0] != '|':
                continue

            crt_parsing = line.strip().split('|')
            crt_parsing = [x.strip() for x in crt_parsing]
            if not crt_parsing:
                continue

            if crt_parsing[1] == 'Interval':
                continue

            crt_interval = crt_parsing[1]

            if crt_interval != '':
                interval = parse_interval(crt_interval)
            # print(parse_subject_room_prof(crt_parsing[2], timetable_specs[PROFESORI]))

            idx = 2

            for day in timetable_specs[ZILE]:
                subject, room, prof = parse_subject_room_prof(crt_parsing[idx], initials_to_prof)
                if subject:
                    # ACEEASI SALA ESTE OCUPATA DE 2 MATERII IN ACELASI INTERVAL
                    if room in timetable[day][interval]:
                        print(f'Sala {room} este ocupata de 2 materii in acelasi interval!')
                        raise Exception('Sala ocupata de 2 materii in acelasi interval!')

                    timetable[day][interval][room] = prof, subject 
                else:
                    timetable[day][interval][room] = None
                idx += 1


    return timetable


def check_mandatory_constraints(timetable : {str : {(int, int) : {str : (str, str)}}}, timetable_specs : dict):
    '''
    Se verifică dacă orarul generat respectă cerințele obligatorii pentru a fi un orar valid.
    '''

    constrangeri_incalcate = 0

    acoperire_target = timetable_specs[MATERII]
    
    acoperire_reala = {subject : 0 for subject in acoperire_target}

    ore_profesori = {prof : 0 for prof in timetable_specs[PROFESORI]}

    for day in timetable:
        for interval in timetable[day]:
            profs_in_crt_interval = []
            for room in timetable[day][interval]:
                if timetable[day][interval][room]:
                    prof, subject = timetable[day][interval][room]
                    acoperire_reala[subject] += timetable_specs[SALI][room][CAPACITATE]

                    # PROFESORUL PREDĂ 2 MATERII ÎN ACELAȘI INTERVAL
                    if prof in profs_in_crt_interval:
                        print(f'Profesorul {prof} preda 2 materii in acelasi interval!')
                        constrangeri_incalcate += 1
                    else:
                        profs_in_crt_interval.append(prof)

                    # MATERIA NU SE PREDA IN SALA
                    if subject not in timetable_specs[SALI][room][MATERII]:
                        print(f'Materia {subject} nu se preda în sala {room}!')
                        constrangeri_incalcate += 1

                    # PROFESORUL NU PREDA MATERIA
                    if subject not in timetable_specs[PROFESORI][prof][MATERII]:
                        print(f'Profesorul {prof} nu poate preda materia {subject}!')
                        constrangeri_incalcate += 1

                    ore_profesori[prof] += 1

    # CONDITIA DE ACOPERIRE
    for subject in acoperire_target:
        if acoperire_reala[subject] < acoperire_target[subject]:
            print(f'Materia {subject} nu are acoperirea necesară!')
            constrangeri_incalcate += 1

    # CONDITIA DE MAXIM 7 ORE PE SĂPTĂMÂNĂ
    for prof in ore_profesori:
        if ore_profesori[prof] > 7:
            print(f'Profesorul {prof} tine mai mult de 7 sloturi!')
            constrangeri_incalcate += 1

    return constrangeri_incalcate


def check_optional_constraints(timetable : {str : {(int, int) : {str : (str, str)}}}, timetable_specs : dict):
    '''
    Se verifică dacă orarul generat respectă cerințele profesorilor pentru a fi un orar valid.
    '''

    constrangeri_incalcate = 0

    for prof in timetable_specs[PROFESORI]:
        for const in timetable_specs[PROFESORI][prof][CONSTRANGERI]:
            if const[0] != '!':
                continue
            else:
                const = const[1:]

                if const in timetable_specs[ZILE]:
                    day = const
                    if day in timetable:
                        for interval in timetable[day]:
                            for room in timetable[day][interval]:
                                if timetable[day][interval][room]:
                                    crt_prof, _ = timetable[day][interval][room]
                                    if prof == crt_prof:
                                        print(f'Profesorul {prof} nu dorește să predea în ziua {day}!')
                                        constrangeri_incalcate += 1
                
                elif '-' in const:
                    interval = parse_interval(const)
                    start, end = interval

                    if start != end - 2:
                        intervals = [(i, i + 2) for i in range(start, end, 2)]
                    else:
                        intervals = [(start, end)]


                    for day in timetable:
                        for interval in intervals:
                            if interval in timetable[day]:
                                for room in timetable[day][interval]:
                                    if timetable[day][interval][room]:
                                        crt_prof, _ = timetable[day][interval][room]
                                        if prof == crt_prof:
                                            print(f'Profesorul {prof} nu dorește să predea în intervalul {interval}!')
                                            constrangeri_incalcate += 1

    return constrangeri_incalcate

if __name__ == '__main__':

    
    if len(sys.argv) == 1:
        print('\nSe rulează de exemplu:\n\npython3 check_constraints.py orar_mic_exact\n')
        sys.exit(0)

    if sys.argv[1] == '-h':
        print('\nSe rulează de exemplu:\n\npython3 check_constraints.py orar_mic_exact\n')

    name = sys.argv[1]

    input_name = f'inputs/{name}.yaml'
    output_name = f'outputs/{name}.txt'

    timetable_specs = read_yaml_file(input_name)

    debug_flag = False


    timetable = get_timetable(timetable_specs, output_name, debug_flag)

    if debug_flag:
        print(pretty_print_timetable(timetable, input_name))

    print('\n----------- Constrângeri obligatorii -----------')
    constrangeri_incalcate = check_mandatory_constraints(timetable, timetable_specs)

    print(f'\n=>\nS-au încălcat {constrangeri_incalcate} constrângeri obligatorii!')

    print('\n----------- Constrângeri optionale -----------')
    constrangeri_optionale = check_optional_constraints(timetable, timetable_specs)
    
    print(f'\n=>\nS-au încălcat {constrangeri_optionale} constrângeri optionale!\n')

