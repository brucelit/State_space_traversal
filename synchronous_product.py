from astar_implementation.petrinet import PetriNet, Marking
from pm4py.objects.petri.utils import add_arc_from_to


def construct(pn1, im1, fm1, pn2, im2, fm2, skip):
    """
    Constructs the synchronous product net of two given Petri nets.


    :param pn1: Petri net 1
    :param im1: Initial marking of Petri net 1
    :param fm1: Final marking of Petri net 1
    :param pn2: Petri net 2
    :param im2: Initial marking of Petri net 2
    :param fm2: Final marking of Petri net 2
    :param skip: Symbol to be used as skip

    Returns
    -------
    :return: Synchronous product net and associated marking labels are of the form (a,>>)
    """
    sync_net = PetriNet('synchronous_product_net of %s and %s' % (pn1.name, pn2.name))
    t1_map, p1_map = __copy_into(pn1, sync_net, True, skip)
    t2_map, p2_map = __copy_into(pn2, sync_net, False, skip)
    sync_index = {}
    for t1 in pn1.transitions:
        for t2 in pn2.transitions:
            if t1.label == t2.label:
                # print("t1 order:", t1.order)
                sync = PetriNet.Transition((t1.name, t2.name), (t1.label, t2.label))
                sync_index[sync] = t1.order
                sync_net.transitions.add(sync)
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

    return sync_net, sync_im, sync_fm, sync_index



def __copy_into(source_net, target_net, upper, skip):
    t_map = {}
    p_map = {}
    for t in source_net.transitions:
        name = (t.name, skip) if upper else (skip, t.name)
        label = (t.label, skip) if upper else (skip, t.label)
        t_map[t] = PetriNet.Transition(name, label)
        target_net.transitions.add(t_map[t])

    for p in source_net.places:
        name = (p.name, skip) if upper else (skip, p.name)
        p_map[p] = PetriNet.Place(name)
        target_net.places.add(p_map[p])

    for t in source_net.transitions:
        for a in t.in_arcs:
            add_arc_from_to(p_map[a.source], t_map[t], target_net)
        for a in t.out_arcs:
            add_arc_from_to(t_map[t], p_map[a.target], target_net)

    return t_map, p_map
