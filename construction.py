from collections import Counter

from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils.petri_utils import add_arc_from_to
from pm4py.objects.petri_net import properties

# compute the split point before search
def precompute_forward(trace_lst, ic):
    dict_l = ic.rule_l
    dict_r = ic.rule_r
    i = 1
    rule_lst = [i for i in range(0, len(dict_r))]
    violate_lst = {}
    while i < len(trace_lst):
        trace_prefix = Counter(trace_lst[0:i])
        for j in rule_lst:
            count_l = 0
            count_r = 0
            for k, v in trace_prefix.items():
                if k in dict_l[j]:
                    count_l += v
                if k in dict_r[j]:
                    count_r += v
            if count_l < count_r:
                rule_lst.remove(j)
                violate_lst[trace_lst[0:i][-1]] = i
                continue
        i += 1
    lst = list(violate_lst.values())
    lst2 = []
    for i in range(len(lst) - 1):
        if lst[i] + 1 == lst[i + 1]:
            lst2.append(lst[i])
        i += 1
    v2 = {key: val for key, val in violate_lst.items() if val not in lst2}
    return v2

# compute the split point before search
def precompute_backward(trace_lst, ic):
    dict_r = ic.rule_l
    dict_l = ic.rule_r
    i = 1
    rule_lst = [i for i in range(0, len(dict_r))]
    violate_lst = {}
    while i < len(trace_lst):
        trace_prefix = Counter(trace_lst[0:i])
        for j in rule_lst:
            count_l = 0
            count_r = 0
            for k, v in trace_prefix.items():
                if k in dict_l[j]:
                    count_l += v
                if k in dict_r[j]:
                    count_r += v
            if count_l < count_r:
                rule_lst.remove(j)
                violate_lst[trace_lst[0:i][-1]] = i
                continue
        i += 1
    lst = list(violate_lst.values())
    lst2 = []
    for i in range(len(lst) - 1):
        if lst[i] + 1 == lst[i + 1]:
            lst2.append(lst[i])
        i += 1
    v2 = {key: val for key, val in violate_lst.items() if val not in lst2}
    return v2


def construct_cost_aware_forward(pn1, im1, fm1, pn2, im2, fm2, skip, pn1_costs, pn2_costs, sync_costs):
    """
    Constructs the synchronous product net of two given Petri nets.


    :param pn1: Petri net 1
    :param im1: Initial marking of Petri net 1
    :param fm1: Final marking of Petri net 1
    :param pn2: Petri net 2
    :param im2: Initial marking of Petri net 2
    :param fm2: Final marking of Petri net 2
    :param skip: Symbol to be used as skip
    :param pn1_costs: dictionary mapping transitions of pn1 to corresponding costs
    :param pn2_costs: dictionary mapping transitions of pn2 to corresponding costs
    :param pn1_costs: dictionary mapping pairs of transitions in pn1 and pn2 to costs
    :param sync_costs: Costs of sync moves

    Returns
    -------
    :return: Synchronous product net and associated marking labels are of the form (a,>>)
    """
    sync_net = PetriNet('forward synchronous_product_net')
    t1_map, p1_map = __copy_into(pn1, sync_net, True, skip)
    t2_map, p2_map = __copy_into(pn2, sync_net, False, skip)
    costs = dict()
    lst_t_pn1 = []
    lst_t_pn2 = []
    for t in pn1.transitions:
        lst_t_pn1.append(t)
    for t in pn2.transitions:
        lst_t_pn2.append(t)
    lst_t_pn1.sort(key=lambda k: k.name)
    lst_t_pn2.sort(key=lambda k: k.name)

    for t1 in lst_t_pn1:
        costs[t1_map[t1]] = pn1_costs[t1]
    for t2 in lst_t_pn2:
        costs[t2_map[t2]] = pn2_costs[t2]
    for t1 in lst_t_pn1:
        for t2 in lst_t_pn2:
            if t1.label == t2.label:
                sync = PetriNet.Transition((t1.name, t2.name), (t1.label, t2.label))
                sync_net.transitions.add(sync)
                costs[sync] = sync_costs[(t1, t2)]
                # copy the properties of the transitions inside the transition of the sync net
                for p1 in t1.properties:
                    sync.properties[p1] = t1.properties[p1]
                for p2 in t2.properties:
                    sync.properties[p2] = t2.properties[p2]
                for a in t1.in_arcs:
                    add_arc_from_to(p1_map[a.source], sync, sync_net)
                for a in t2.in_arcs:
                    add_arc_from_to(p2_map[a.source], sync, sync_net)
                for a in t1.out_arcs:
                    add_arc_from_to(sync, p1_map[a.target], sync_net)
                for a in t2.out_arcs:
                    add_arc_from_to(sync, p2_map[a.target], sync_net)

    sync_im = Marking()
    sync_fm = Marking()
    for p in im1:
        sync_im[p1_map[p]] = im1[p]
    for p in im2:
        sync_im[p2_map[p]] = im2[p]
    for p in fm1:
        sync_fm[p1_map[p]] = fm1[p]
    for p in fm2:
        sync_fm[p2_map[p]] = fm2[p]

    # update 06/02/2021: to distinguish the sync nets that are output of this method, put a property in the sync net
    sync_net.properties[properties.IS_SYNC_NET] = True
    # print("costs", costs)
    return sync_net, sync_im, sync_fm, costs


def construct_cost_aware_backward(pn1, im1, fm1, pn2, im2, fm2, skip, pn1_costs, pn2_costs, sync_costs):
    """
    Constructs the synchronous product net of two given Petri nets.


    :param pn1: Petri net 1
    :param im1: Initial marking of Petri net 1
    :param fm1: Final marking of Petri net 1
    :param pn2: Petri net 2
    :param im2: Initial marking of Petri net 2
    :param fm2: Final marking of Petri net 2
    :param skip: Symbol to be used as skip
    :param pn1_costs: dictionary mapping transitions of pn1 to corresponding costs
    :param pn2_costs: dictionary mapping transitions of pn2 to corresponding costs
    :param pn1_costs: dictionary mapping pairs of transitions in pn1 and pn2 to costs
    :param sync_costs: Costs of sync moves

    Returns
    -------
    :return: Synchronous product net and associated marking labels are of the form (a,>>)
    """
    sync_net = PetriNet('backward synchronous_product_net')
    t1_map, p1_map = __copy_into2(pn1, sync_net, True, skip)
    t2_map, p2_map = __copy_into2(pn2, sync_net, False, skip)
    costs = dict()

    for t1 in pn1.transitions:
        costs[t1_map[t1]] = pn1_costs[t1]
    for t2 in pn2.transitions:
        costs[t2_map[t2]] = pn2_costs[t2]

    for t1 in pn1.transitions:
        for t2 in pn2.transitions:
            if t1.label == t2.label:
                sync = PetriNet.Transition((t1.name, t2.name), (t1.label, t2.label))
                sync_net.transitions.add(sync)
                costs[sync] = sync_costs[(t1, t2)]
                # copy the properties of the transitions inside the transition of the sync net
                for p1 in t1.properties:
                    sync.properties[p1] = t1.properties[p1]
                for p2 in t2.properties:
                    sync.properties[p2] = t2.properties[p2]
                for a in t1.in_arcs:
                    add_arc_from_to(sync, p1_map[a.source], sync_net)
                for a in t2.in_arcs:
                    add_arc_from_to(sync, p2_map[a.source], sync_net)
                for a in t1.out_arcs:
                    add_arc_from_to(p1_map[a.target], sync, sync_net)
                for a in t2.out_arcs:
                    add_arc_from_to(p2_map[a.target], sync, sync_net)
    sync_im = Marking()
    sync_fm = Marking()
    for p in fm1:
        sync_im[p1_map[p]] = fm1[p]
    for p in fm2:
        sync_im[p2_map[p]] = fm2[p]
    for p in im1:
        sync_fm[p1_map[p]] = im1[p]
    for p in im2:
        sync_fm[p2_map[p]] = im2[p]

    # update 06/02/2021: to distinguish the sync nets that are output of this method, put a property in the sync net
    sync_net.properties[properties.IS_SYNC_NET] = True
    return sync_net, sync_im, sync_fm, costs


def __copy_into(source_net, target_net, upper, skip):
    t_map = {}
    p_map = {}
    lst_t = []
    lst_p = []
    for p in source_net.places:
        lst_p.append(p)
    for t in source_net.transitions:
        lst_t.append(t)
    lst_t.sort(key=lambda k: k.name)
    lst_p.sort(key=lambda k: k.name)

    for t in lst_t:
        name = (t.name, skip) if upper else (skip, t.name)
        label = (t.label, skip) if upper else (skip, t.label)
        t_map[t] = PetriNet.Transition(name, label)
        if properties.TRACE_NET_TRANS_INDEX in t.properties:
            # 16/02/2021: copy the index property from the transition of the trace net
            t_map[t].properties[properties.TRACE_NET_TRANS_INDEX] = t.properties[properties.TRACE_NET_TRANS_INDEX]
        target_net.transitions.add(t_map[t])

    for p in lst_p:
        name = (p.name, skip) if upper else (skip, p.name)
        p_map[p] = PetriNet.Place(name)
        if properties.TRACE_NET_PLACE_INDEX in p.properties:
            # 16/02/2021: copy the index property from the place of the trace net
            p_map[p].properties[properties.TRACE_NET_PLACE_INDEX] = p.properties[properties.TRACE_NET_PLACE_INDEX]
        target_net.places.add(p_map[p])

    for t in lst_t:
        for a in t.in_arcs:
            add_arc_from_to(p_map[a.source], t_map[t], target_net)
        for a in t.out_arcs:
            add_arc_from_to(t_map[t], p_map[a.target], target_net)
    # print("t map", t_map)
    # print("p map", p_map)
    return t_map, p_map


def __copy_into2(source_net, target_net, upper, skip):
    t_map = {}
    p_map = {}
    for t in source_net.transitions:
        name = (t.name, skip) if upper else (skip, t.name)
        label = (t.label, skip) if upper else (skip, t.label)
        t_map[t] = PetriNet.Transition(name, label)
        if properties.TRACE_NET_TRANS_INDEX in t.properties:
            # 16/02/2021: copy the index property from the transition of the trace net
            t_map[t].properties[properties.TRACE_NET_TRANS_INDEX] = t.properties[properties.TRACE_NET_TRANS_INDEX]
        target_net.transitions.add(t_map[t])


    for p in source_net.places:
        name = (p.name, skip) if upper else (skip, p.name)
        p_map[p] = PetriNet.Place(name)
        if properties.TRACE_NET_PLACE_INDEX in p.properties:
            # 16/02/2021: copy the index property from the place of the trace net
            p_map[p].properties[properties.TRACE_NET_PLACE_INDEX] = p.properties[properties.TRACE_NET_PLACE_INDEX]
        target_net.places.add(p_map[p])

    for t in source_net.transitions:
        for a in t.in_arcs:
            add_arc_from_to(t_map[t], p_map[a.source], target_net)
        for a in t.out_arcs:
            add_arc_from_to(p_map[a.target], t_map[t], target_net)
    return t_map, p_map