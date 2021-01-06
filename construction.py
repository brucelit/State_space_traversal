from pm4py.objects.petri import utils as petri_utils

from astar_implementation.petrinet import PetriNet, Marking
from pm4py.visualization.petrinet import visualizer


def construct_model_net():
    model_net = PetriNet("model_net")
    p1 = PetriNet.Place("p1")
    p2 = PetriNet.Place("p2")
    p3 = PetriNet.Place("p3")
    p4 = PetriNet.Place("p4")
    p5 = PetriNet.Place("p5")
    p6 = PetriNet.Place("p6")
    p7 = PetriNet.Place("p7")
    model_net.places.add(p1)
    model_net.places.add(p2)
    model_net.places.add(p3)
    model_net.places.add(p4)
    model_net.places.add(p5)
    model_net.places.add(p6)
    model_net.places.add(p7)

    # add transitions for model
    t1 = PetriNet.Transition("t1", "a")
    t2 = PetriNet.Transition("t2", "b")
    t3 = PetriNet.Transition("t3", "c")
    t4 = PetriNet.Transition("t4", "d")
    t5 = PetriNet.Transition("t5", "e")
    t6 = PetriNet.Transition("t6", "e")
    t7 = PetriNet.Transition("t7", chr(964))
    t8 = PetriNet.Transition("t8", "g")
    t9 = PetriNet.Transition("t9", "h")
    model_net.transitions.add(t1)
    model_net.transitions.add(t2)
    model_net.transitions.add(t3)
    model_net.transitions.add(t4)
    model_net.transitions.add(t5)
    model_net.transitions.add(t6)
    model_net.transitions.add(t7)
    model_net.transitions.add(t8)
    model_net.transitions.add(t9)

    # add arcs for model
    petri_utils.add_arc_from_to(p1, t1, model_net)
    petri_utils.add_arc_from_to(t1, p2, model_net)
    petri_utils.add_arc_from_to(t1, p3, model_net)
    petri_utils.add_arc_from_to(p2, t2, model_net)
    petri_utils.add_arc_from_to(p2, t3, model_net)
    petri_utils.add_arc_from_to(p3, t4, model_net)
    petri_utils.add_arc_from_to(p3, t5, model_net)
    petri_utils.add_arc_from_to(t2, p4, model_net)
    petri_utils.add_arc_from_to(t3, p4, model_net)
    petri_utils.add_arc_from_to(p4, t5, model_net)
    petri_utils.add_arc_from_to(p4, t6, model_net)
    petri_utils.add_arc_from_to(t4, p5, model_net)
    petri_utils.add_arc_from_to(t5, p6, model_net)
    petri_utils.add_arc_from_to(t6, p6, model_net)
    petri_utils.add_arc_from_to(p5, t6, model_net)
    petri_utils.add_arc_from_to(p6, t7, model_net)
    petri_utils.add_arc_from_to(p6, t8, model_net)
    petri_utils.add_arc_from_to(p6, t9, model_net)
    petri_utils.add_arc_from_to(t7, p2, model_net)
    petri_utils.add_arc_from_to(t7, p3, model_net)
    petri_utils.add_arc_from_to(t8, p7, model_net)
    petri_utils.add_arc_from_to(t9, p7, model_net)

    # add marking for model
    model_im = Marking()
    model_im[p1] = 1
    model_fm = Marking()
    model_fm[p7] = 1
    # gviz = visualizer.apply(model_net, model_im, model_fm)
    # visualizer.view(gviz)
    return model_net, model_im, model_fm


def construct_trace(trace):
    trace_net = PetriNet("trace_net")
    trace_to_lst = list(trace)
    len1 = len(trace_to_lst)
    trace_im = Marking()
    trace_fm = Marking()
    for i in range(1, len1+2):
        p_i = str('p')+str(i)
        p = PetriNet.Place(p_i)
        trace_net.places.add(p)
        if i == 1:
            trace_im[p] = 1
        else:
            petri_utils.add_arc_from_to(t, p, trace_net)
        if i < len1+1:
            t_i = str('t') + str(i)
            t = PetriNet.Transition(t_i, trace_to_lst[i - 1], order=i)
            trace_net.transitions.add(t)
            petri_utils.add_arc_from_to(p, t, trace_net)
    trace_fm[p] = 1
    # gviz2 = visualizer.apply(trace_net, trace_im, trace_fm)
    # visualizer.view(gviz2)
    return trace_net, trace_im, trace_fm



def construct_trace_ccc(trace_to_lst):
    trace_net = PetriNet("trace_net")
    len1 = len(trace_to_lst)
    trace_im = Marking()
    trace_fm = Marking()
    for i in range(1, len1+2):
        p_i = str('p')+str(i)
        p = PetriNet.Place(p_i)
        trace_net.places.add(p)
        if i == 1:
            trace_im[p] = 1
        else:
            petri_utils.add_arc_from_to(t, p, trace_net)
        if i < len1+1:
            t_i = str('t') + str(i)
            t = PetriNet.Transition(t_i, trace_to_lst[i - 1], order=i)
            trace_net.transitions.add(t)
            petri_utils.add_arc_from_to(p, t, trace_net)
    trace_fm[p] = 1
    # gviz2 = visualizer.apply(trace_net, trace_im, trace_fm)
    # visualizer.view(gviz2)
    return trace_net, trace_im, trace_fm