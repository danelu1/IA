import yaml
import argparse
import sys

##################### MACROURI #####################
INTERVALE = 'Intervale'
ZILE = 'Zile'
MATERII = 'Materii'
PROFESORI = 'Profesori'
SALI = 'Sali'

def read_yaml_file(file_path : str) -> dict:
    '''
    Citeste un fișier yaml și returnează conținutul său sub formă de dicționar
    '''
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def acces_yaml_attributes(yaml_dict : dict):
    '''
    Primește un dicționar yaml și afișează datele referitoare la atributele sale
    '''

    print('Zilele din orar sunt:', yaml_dict[ZILE])
    print()
    print('Intervalele orarului sunt:', yaml_dict[INTERVALE])
    print()
    print('Materiile sunt:', yaml_dict[MATERII])
    print()
    print('Profesorii sunt:', end=' ')
    print(*list(yaml_dict[PROFESORI].keys()), sep=', ')
    print()
    print('Sălile sunt:', end=' ')
    print(*list(yaml_dict[SALI].keys()), sep=', ')


def get_profs_initials(profs : list) -> dict:
    '''
    Primește o listă de profesori

    Returnează două dicționare:
    - unul care are numele profesorilor drept chei și drept valori prescurtările lor (prof_to_initials[prof] = initiale)
    - unul care are prescurtările profesorilor drept chei și drept valori numele lor (initials_to_prof[initiale] = prof)
    '''

    initials_to_prof = {}
    prof_to_initials = {}
    initials_count = {}

    for prof in profs:
        name_components = prof.split(' ')
        initials = name_components[0][0] + name_components[1][0]
        
        if initials in initials_count:
            initials_count[initials] += 1
            initials += str(initials_count[initials])
        else:
            initials_count[initials] = 1
        
        initials_to_prof[initials] = prof
        prof_to_initials[prof] = initials
        
    return prof_to_initials, initials_to_prof


def allign_string_with_spaces(s : str, max_len : int, allignment_type : str = 'center') -> str:
    '''
    Primește un string și un număr întreg

    Returnează string-ul dat, completat cu spații până la lungimea dată
    '''

    len_str = len(s)

    if len_str >= max_len:
        raise ValueError('Lungimea string-ului este mai mare decât lungimea maximă dată')
    

    if allignment_type == 'left':
        s = 6 * ' ' + s
        s += (max_len - len(s)) * ' '

    elif allignment_type == 'center':
        if len_str % 2 == 1:
            s = ' ' + s
        s = s.center(max_len, ' ')

    return s


def pretty_print_timetable_aux_zile(timetable : {str : {(int, int) : {str : (str, str)}}}, input_path : str) -> str:
    '''
    Primește un dicționar ce are chei zilele, cu valori dicționare de intervale reprezentate ca tupluri de int-uri, cu valori dicționare de săli, cu valori tupluri (profesor, materie)

    Returnează un string formatat să arate asemenea unui tabel excel cu zilele pe linii, intervalele pe coloane și în intersecția acestora, ferestrele de 2 ore cu materiile alocate în fiecare sală fiecărui profesor
    '''

    max_len = 30

    profs = read_yaml_file(input_path)[PROFESORI].keys()
    profs_to_initials, _ = get_profs_initials(profs)

    table_str = '|           Interval           |             Luni             |             Marti            |           Miercuri           |              Joi             |            Vineri            |\n'

    no_classes = len(timetable['Luni'][(8, 10)])

    first_line_len = 187
    delim = '-' * first_line_len + '\n'
    table_str = table_str + delim
    
    for interval in timetable['Luni']:
        s_interval = '|'
        
        crt_str = allign_string_with_spaces(f'{interval[0]} - {interval[1]}', max_len, 'center')

        s_interval += crt_str

        for class_idx in range(no_classes):
            if class_idx != 0:
                s_interval += f'|{30 * " "}'

            for day in timetable:
                classes = timetable[day][interval]
                classroom = list(classes.keys())[class_idx]

                s_interval += '|'

                if not classes[classroom]:
                    s_interval += allign_string_with_spaces(f'{classroom} - goala', max_len, 'left')
                else:
                    prof, subject = classes[classroom]
                    s_interval += allign_string_with_spaces(f'{subject} : ({classroom} - {profs_to_initials[prof]})', max_len, 'left')
            
            s_interval += '|\n'
        table_str += s_interval + delim

    return table_str

def pretty_print_timetable_aux_intervale(timetable : {(int, int) : {str : {str : (str, str)}}}, input_path : str) -> str:
    '''
    Primește un dicționar de intervale reprezentate ca tupluri de int-uri, cu valori dicționare de zile, cu valori dicționare de săli, cu valori tupluri (profesor, materie)

    Returnează un string formatat să arate asemenea unui tabel excel cu zilele pe linii, intervalele pe coloane și în intersecția acestora, ferestrele de 2 ore cu materiile alocate în fiecare sală fiecărui profesor
    '''

    max_len = 30

    profs = read_yaml_file(input_path)[PROFESORI].keys()
    profs_to_initials, _ = get_profs_initials(profs)

    table_str = '|           Interval           |             Luni             |             Marti            |           Miercuri           |              Joi             |            Vineri            |\n'

    no_classes = len(timetable[(8, 10)]['Luni'])

    first_line_len = 187
    delim = '-' * first_line_len + '\n'
    table_str = table_str + delim
    
    for interval in timetable:
        s_interval = '|' + allign_string_with_spaces(f'{interval[0]} - {interval[1]}', max_len, 'center')

        for class_idx in range(no_classes):
            if class_idx != 0:
                s_interval += '|'

            for day in timetable[interval]:
                classes = timetable[interval][day]
                classroom = list(classes.keys())[class_idx]

                s_interval += '|'

                if not classes[classroom]:
                    s_interval += allign_string_with_spaces(f'{classroom} - goala', max_len, 'left')
                else:
                    prof, subject = classes[classroom]
                    s_interval += allign_string_with_spaces(f'{subject} : ({classroom} - {profs_to_initials[prof]})', max_len, 'left')
            
            s_interval += '|\n'
        table_str += s_interval + delim

    return table_str

def pretty_print_timetable(timetable : dict, input_path : str) -> str:
    '''
    Poate primi fie un dictionar de zile conținând dicționare de intervale conținând dicționare de săli cu tupluri (profesor, materie)
    fie un dictionar de intervale conținând dictionare de zile conținând dicționare de săli cu tupluri (profesor, materie)
    
    Pentru cazul în care o sală nu este ocupată la un moment de timp, se așteaptă 'None' în valoare, în loc de tuplu
    '''
    if 'Luni' in timetable:
        return pretty_print_timetable_aux_zile(timetable, input_path)
    else:
        return pretty_print_timetable_aux_intervale(timetable, input_path)


if __name__ == '__main__':
    filename = f'inputs/orar_mic_exact.yaml'

    timetable_specs = read_yaml_file(filename)

    acces_yaml_attributes(timetable_specs)

    # print(pretty_print_timetable(timetable, filename))
