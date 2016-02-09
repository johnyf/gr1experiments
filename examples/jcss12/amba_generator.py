#!/usr/bin/env python
"""Generate AMBA AHB specifications for given number of masters.

Translated and adapted from Perl original distributed with Anzu.
    https://www.iaik.tugraz.at/content/research/opensource/anzu/#download
"""
import argparse
import math
from omega.logic.syntax import conj


def build_state_str(state_name, op, num_states, value,
                    padd_value='0', add_next=''):
    result = ''
    binary = bin(value).lstrip('-0b').zfill(1)[::-1]
    for j in xrange(num_states):
        if result != '':
            result += op
        bin_val = padd_value
        if j < len(binary):
            bin_val = binary[j]
        result += '{add_next}({state_name}{j} = {bin_val})'.format(
            add_next=add_next, state_name=state_name, j=j, bin_val=bin_val)
    return result


def build_hmaster_str(master_bits, value):
    return build_state_str('hmaster', ' & ', master_bits, value)


def generate_spec(num_masters, use_ba):
    # init
    master_bits = int(math.ceil(math.log(num_masters) / math.log(2.0)))
    master_bits_plus_one = math.ceil(math.log(num_masters + 1) / math.log(2))
    assert master_bits > 0, master_bits
    assert master_bits_plus_one > 0, master_bits_plus_one
    env_initial = list()
    sys_initial = ''
    env_transitions = ''
    sys_transitions = list()
    env_fairness = ''
    sys_fairness = ''
    input_vars = list()
    output_vars = list()
    ###############################################
    # ENV_INITIAL and INPUT_VARIABLES
    ###############################################
    env_initial.append('hready = 0')
    input_vars += ['hready', 'hburst0', 'hburst1']
    for i in xrange(num_masters):
        s = 'hbusreq{i} = 0'.format(i=i)
        env_initial.append(s)
        s = 'hlock{i} = 0'.format(i=i)
        env_initial.append(s)
        s = 'hbusreq{i}'.format(i=i)
        input_vars.append(s)
        s = 'hlock{i}'.format(i=i)
        input_vars.append(s)
    env_initial.append('hburst0 = 0')
    env_initial.append('hburst1 = 0')
    ###############################################
    # ENV_TRANSITION
    ###############################################
    for i in xrange(num_masters):
        # env_transitions += "#Assumption 3:\n"
        env_transitions += "[]( hlock{i} = 1 -> hbusreq{i} = 1 ) & \n".format(i=i)
    ###############################################
    # ENV_FAIRNESS
    ###############################################
    env_fairness += (
        # "# Assumption 1: \n"
        "[](<>(stateA1_1 = 0)) & \n"
        # "\n# Assumption 2:\n"
        "[](<>(hready = 1))\n")
    ###############################################
    # SYS_INITIAL + OUTPUT_VARIABLES
    ###############################################
    for i in xrange(master_bits):
        sys_initial += 'hmaster{i} = 0 & \n'.format(i=i)
        output_vars.append('hmaster{i}'.format(i=i))
    output_vars += [
        "hmastlock", "start", "locked", "decide", 'hgrant0',
        "busreq", "stateA1_0", "stateA1_1", "stateG2",
        "stateG3_0", "stateG3_1", "stateG3_2"]
    c = [
        "hmastlock = 0",
        "start = 1",
        "decide = 1",
        "locked = 0",
        "hgrant0 = 1"]
    sys_initial += '&\n'.join(c) + '&\n'
    for i in xrange(1, num_masters):
        sys_initial += "hgrant{i} = 0 & \n".format(i=i)
        var = 'hgrant{i}'.format(i=i)
        output_vars.append(var)
    # busreq = hbusreq[hmaster]
    sys_initial += (
        "busreq=0 & \n"
        # Assumption 1:
        "stateA1_0 = 0 & \n"
        "stateA1_1 = 0 & \n"
        # Guarantee 2:
        "stateG2 = 0 & \n"
        # Guarantee 3:
        "stateG3_0 = 0 & \n"
        "stateG3_1 = 0 & \n"
        "stateG3_2 = 0 & \n")
    # Guarantee 10:
    for i in xrange(1, num_masters):
        sys_initial += "stateG10_{i} = 0 & \n".format(i=i)
        var = 'stateG10_{i}'.format(i=i)
        output_vars.append(var)
    ###############################################
    # SYS_TRANSITION
    ###############################################
    # busreq = hbusreq[hmaster]
    for i in xrange(num_masters):
        hmaster = build_hmaster_str(master_bits, i)
        hmaster_X = build_state_str("hmaster", " & ", master_bits, i, 0, 'X')
        sys_transitions.append((
            "[]({hmaster} -> (hbusreq{i} = 0 <-> busreq=0))").format(
                i=i, hmaster=hmaster))
    # Assumption 1:
    # state 00
    sys_transitions.append(
        # "# Assumption 1:\n"
        "[](((stateA1_1 = 0) & (stateA1_0 = 0) & "
        "((hmastlock = 0) | (hburst0 = 1) | (hburst1 = 1))) ->\n"
        " X((stateA1_1 = 0) & (stateA1_0 = 0))) & \n"
        "[](((stateA1_1 = 0) & (stateA1_0 = 0) & "
        " (hmastlock = 1) & (hburst0 = 0) & (hburst1 = 0)) ->\n"
        " X((stateA1_1 = 1) & (stateA1_0 = 0))) & \n"
        # state 10
        "[](((stateA1_1 = 1) & (stateA1_0 = 0) & (busreq = 1)) ->\n"
        " X((stateA1_1 = 1) & (stateA1_0 = 0))) & \n"
        "[](((stateA1_1 = 1) & (stateA1_0 = 0) & (busreq = 0) & "
        "((hmastlock = 0) | (hburst0 = 1) | (hburst1 = 1))) ->\n"
        " X((stateA1_1 = 0) & (stateA1_0 = 0))) & \n"
        "[](((stateA1_1 = 1) & (stateA1_0 = 0) & (busreq = 0) & "
        " (hmastlock = 1) & (hburst0 = 0) & (hburst1 = 0)) ->\n"
        " X((stateA1_1 = 0) & (stateA1_0 = 1))) & \n"
        # state 01
        "[](((stateA1_1 = 0) & (stateA1_0 = 1) & (busreq = 1)) ->\n"
        " X((stateA1_1 = 1) & (stateA1_0 = 0))) & \n"
        "[](((stateA1_1 = 0) & (stateA1_0 = 1) & "
        " (hmastlock = 1) & (hburst0 = 0) & (hburst1 = 0)) ->\n"
        " X((stateA1_1 = 1) & (stateA1_0 = 0))) & \n"
        "[](((stateA1_1 = 0) & (stateA1_0 = 1) & (busreq = 0) & "
        "((hmastlock = 0) | (hburst0 = 1) | (hburst1 = 1))) ->\n"
        " X((stateA1_1 = 0) & (stateA1_0 = 0))) & \n"
        # Guarantee 1:
        # sys_transitions += "\n# Guarantee 1:\n"
        "[]((hready = 0) -> X(start = 0)) & \n"
        # Guarantee 2:
        # sys_transitions += "\n# Guarantee 2:\n"
        "[](((stateG2 = 0) & "
        "((hmastlock = 0) | (start = 0) | "

        "(hburst0 = 1) | (hburst1 = 1))) -> "
        "X(stateG2 = 0)) & \n"

        "[](((stateG2 = 0) & "
        " (hmastlock = 1) & (start = 1) & "

        "(hburst0 = 0) & (hburst1 = 0))  -> "
        "X(stateG2 = 1)) & \n"

        "[](((stateG2 = 1) & (start = 0) & (busreq = 1)) -> "
        "X(stateG2 = 1)) & \n"
        "[](((stateG2 = 1) & (start = 1)) -> false) & \n"
        "[](((stateG2 = 1) & (start = 0) & (busreq = 0)) -> "
        "X(stateG2 = 0)) & \n"
        # Guarantee 3:
        # sys_transitions += "\n# Guarantee 3:\n"
        '[](((stateG3_0 = 0) & (stateG3_1 = 0) & (stateG3_2 = 0) & \n'
        '  ((hmastlock = 0) | (start = 0) | ((hburst0 = 1) | (hburst1 = 0)))) ->\n'
        '  (X(stateG3_0 = 0) & X(stateG3_1 = 0) & X(stateG3_2 = 0))) &\n'
        '[](((stateG3_0 = 0) & (stateG3_1 = 0) & (stateG3_2 = 0) & \n'
        '  ((hmastlock = 1) & (start = 1) & '
        '((hburst0 = 0) & (hburst1 = 1)) & (hready = 0))) -> \n'
        '   (X(stateG3_0 = 1) & X(stateG3_1 = 0) & X(stateG3_2 = 0))) &\n'
        '[](((stateG3_0 = 0) & (stateG3_1 = 0) & (stateG3_2 = 0) & \n'
        '  ((hmastlock = 1) & (start = 1) & '
        '((hburst0 = 0) & (hburst1 = 1)) & (hready = 1))) -> \n'
        '   (X(stateG3_0 = 0) & X(stateG3_1 = 1) & X(stateG3_2 = 0))) &\n'
        ' \n'
        '[](((stateG3_0 = 1) & (stateG3_1 = 0) & '
        '(stateG3_2 = 0) & ((start = 0) & (hready = 0))) -> \n'
        '   (X(stateG3_0 = 1) & X(stateG3_1 = 0) & X(stateG3_2 = 0))) &\n'
        '[](((stateG3_0 = 1) & (stateG3_1 = 0) & '
        '(stateG3_2 = 0) & ((start = 0) & (hready = 1))) -> \n'
        '   (X(stateG3_0 = 0) & X(stateG3_1 = 1) & X(stateG3_2 = 0))) &\n'
        '\n'
        '[](((stateG3_0 = 1) & (stateG3_1 = 0) & '
        '(stateG3_2 = 0) & ((start = 1))) -> false) &\n'
        '\n'
        ' \n'
        '[](((stateG3_0 = 0) & (stateG3_1 = 1) & '
        '(stateG3_2 = 0) & ((start = 0) & (hready = 0))) -> \n'
        '   (X(stateG3_0 = 0) & X(stateG3_1 = 1) & X(stateG3_2 = 0))) &\n'
        '[](((stateG3_0 = 0) & (stateG3_1 = 1) & '
        '(stateG3_2 = 0) & ((start = 0) & (hready = 1))) -> \n'
        '   (X(stateG3_0 = 1) & X(stateG3_1 = 1) & X(stateG3_2 = 0))) &\n'
        '[](((stateG3_0 = 0) & (stateG3_1 = 1) & '
        '(stateG3_2 = 0) & ((start = 1))) -> false) &\n'
        ' \n'
        '[](((stateG3_0 = 1) & (stateG3_1 = 1) & '
        '(stateG3_2 = 0) & ((start = 0) & (hready = 0))) -> \n'
        '   (X(stateG3_0 = 1) & X(stateG3_1 = 1) & X(stateG3_2 = 0))) &\n'
        '[](((stateG3_0 = 1) & (stateG3_1 = 1) & '
        '(stateG3_2 = 0) & ((start = 0) & (hready = 1))) -> \n'
        '   (X(stateG3_0 = 0) & X(stateG3_1 = 0) & X(stateG3_2 = 1))) &\n'
        '[](((stateG3_0 = 1) & (stateG3_1 = 1) & '
        '(stateG3_2 = 0) & ((start = 1))) -> false) &\n'
        ' \n'
        '[](((stateG3_0 = 0) & (stateG3_1 = 0) & '
        '(stateG3_2 = 1) & ((start = 0) & (hready = 0))) -> \n'
        '   (X(stateG3_0 = 0) & X(stateG3_1 = 0) & X(stateG3_2 = 1))) &\n'
        '[](((stateG3_0 = 0) & (stateG3_1 = 0) & '
        '(stateG3_2 = 1) & ((start = 0) & (hready = 1))) -> \n'
        '   (X(stateG3_0 = 0) & X(stateG3_1 = 0) & X(stateG3_2 = 0))) & \n'
        '\n'
        '[](((stateG3_0 = 0) & (stateG3_1 = 0) & '
        '(stateG3_2 = 1) & ((start = 1))) -> false)')
    # Guarantee 4 and 5:
    # sys_transitions += "\n # Guarantee 4 and 5:\n"
    for i in xrange(num_masters):
        hmaster_X = build_state_str("hmaster", " & ", master_bits, i, 0, 'X')
        # '#  Master {i}:\n'.format(i=i)
        s = "[]((hready = 1) -> ((hgrant{i} = 1) <-> ({hmaster_X})))".format(
                i=i, hmaster_X=hmaster_X)
        sys_transitions.append(s)

    sys_transitions.append(
        # "#  HMASTLOCK:\n"
        "[]((hready = 1) -> (locked = 0 <-> X(hmastlock = 0)))")
    # Guarantee 6.1:
    # FIXME: It would be sufficient to have one formula for each bit of hmaster
    # sys_transitions += "\n# Guarantee 6.1:\n"
    for i in xrange(num_masters):
        hmaster = build_hmaster_str(master_bits, i)
        hmaster_X = build_state_str("hmaster", " & ", master_bits, i, 0, 'X')
        # sys_transitions += '#  Master {i}:\n'.format(i=i)
        sys_transitions.append(
            "[](X(start = 0) -> ((" + hmaster + ") <-> (" +
            hmaster_X + ")))")
    # Guarantee 6.2:
    sys_transitions.append(
        # "\n# Guarantee 6.2:\n"
        "[](((X(start = 0))) -> ((hmastlock = 1) <-> X(hmastlock = 1)))")
    # Guarantee 7:
    # FIXME: formula can be written as
    # G((decide=1  &  X(hgrant{i}=1))-> (hlock{i}=1 <-> X(locked=1)))
    # sys_transitions += "\n# Guarantee 7:\n"
    norequest = list()
    for i in xrange(num_masters):
        s = ('[]((decide = 1 & hlock{i} = 1 & X(hgrant{i} = 1))->'
             'X(locked = 1))').format(i=i)
        sys_transitions.append(s)
        s = ('[]((decide = 1 & hlock{i} = 0 & X(hgrant{i} = 1))->'
             'X(locked = 0))').format(i=i)
        sys_transitions.append(s)
        s = 'hbusreq{i} = 0'.format(i=i)
        norequest.append(s)
    # Guarantee 8:
    # MW: this formula changes with respect to the number of grant signals
    # sys_transitions += "\n# Guarantee 8:\n"
    tmp_g8 = ''
    for i in xrange(num_masters):
        sys_transitions.append((
            '[]((decide = 0) -> (((hgrant{i} = 0)'
            '<-> X(hgrant{i} = 0))))').format(i=i))
    sys_transitions.append('[]((decide = 0)->(locked = 0 <-> X(locked = 0)))')
    # Guarantee 10:
    # sys_transitions += "\n#Guarantee 10:\n"
    for i in xrange(1, num_masters):
        hmaster = build_hmaster_str(master_bits, i)
        # sys_transitions += "#  Master " + i + ":\n"
        sys_transitions.append((
            '[](((stateG10_{i} = 0) & (((hgrant{i} = 1) |'
            '(hbusreq{i} = 1)))) -> X(stateG10_{i} = 0)) & \n'
            '[](((stateG10_{i} = 0) & ((hgrant{i} = 0) & '
            '(hbusreq{i} = 0))) -> X(stateG10_{i} = 1)) & \n'
            '[](((stateG10_{i} = 1) & ((hgrant{i} = 0) & '
            '(hbusreq{i} = 0)))-> X(stateG10_{i} = 1)) & \n'
            '[](((stateG10_{i} = 1) & (((hgrant{i} = 1)) & '
            '(hbusreq{i} = 0))) -> false) & \n'
            '[](((stateG10_{i} = 1) & (hbusreq{i} = 1)) -> '
            'X(stateG10_{i} = 0))').format(i=i))
    sys_transitions.append(
        # "#default master\n"
        '[]((decide=1  & {norequest}) -> X(hgrant0=1))'.format(
            norequest=conj(norequest, sep='\n')))
    ###############################################
    # SYS_FAIRNESS
    ###############################################
    # Guarantee 2:
    sys_fairness += (
        # "\n# Guarantee 2:\n"
        "[](<>(stateG2 = 0)) & \n")
    # Guarantee 3:
    sys_fairness += (
        # "\n# Guarantee 3:\n"
        "[](<>((stateG3_0 = 0)  &  (stateG3_1 = 0)  &  (stateG3_2 = 0))) \n")
    # Guarantee 9:
    # sys_fairness += "\n# Guarantee 9:\n"
    c = list()
    for i in xrange(num_masters):
        c.append((
            "[](<>((" + build_hmaster_str(master_bits, i) +
            ")  |  hbusreq{i} = 0))").format(i=i))
    fairness = '&\n'.join(c)

    template = ('''do
    :: {guard};
        {nested}
    :: else
    od;
    ''')

    ba_fairness = (
        'assert active proctype fairness(){' +
        recurse_fairness(0, num_masters, master_bits, template) +
        '}')

    if not use_ba:
        sys_fairness += ' & ' + fairness
        ba_fairness = ''
    ###############################################
    # dump smv
    prefix = 'amba'
    if use_ba:
        post = '_merged'
    else:
        post = ''
    # dump formula
    fname = '{s}_{i}{post}.txt'.format(s=prefix, i=num_masters, post=post)
    f = open(fname, 'w')
    ltl = [
        'assume ltl {',
        conj(env_initial, sep='\n'),
        ' & ',
        env_transitions,
        env_fairness,
        '}',
        'assert ltl {',
        sys_initial,
        conj(sys_transitions, sep='\n'),
        ' & ',
        sys_fairness,
        '}',
        ba_fairness]
    c = [
        'free env bit ' + ',\n'.join(input_vars) + ';',
        'free sys bit ' + ',\n'.join(output_vars) + ';',
        '\n'.join(ltl)]
    s = '\n'.join(c)
    s = s.replace('=', '==')
    s = s.replace('&', '&&')
    s = s.replace('|', '||')
    f.write(s)
    f.close()


def recurse_fairness(i, num_masters, master_bits, template):
    if i >= num_masters:
        return 'progress: skip; break'
    guard = (
        "((" + build_hmaster_str(master_bits, i) +
        ")  |  hbusreq{i} = 0)").format(i=i)
    nested = recurse_fairness(i + 1, num_masters, master_bits, template)
    s = template.format(guard=guard, nested=nested)
    if i > 1:
        s += '\nbreak;\n'
    return s


def dump_range_of_specs(n, m, use_ba):
    for i in xrange(n, m + 1):
        generate_spec(i, use_ba)


def main():
    description = 'Generator of AMBA AHB bus arbiter spec'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--min', type=int, help='min number of masters')
    parser.add_argument('--max', type=int, help='max number of masters')
    parser.add_argument('--merged', action='store_true', help='use BA')
    args = parser.parse_args()
    n = args.min
    m = args.max
    use_ba = args.merged
    dump_range_of_specs(n, m, use_ba)


if __name__ == '__main__':
    main()
