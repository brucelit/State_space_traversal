from pm4py.objects.petri import utils as petri_utils
from pm4py.objects.petri.petrinet import PetriNet, Marking


def construct_model_net_without_loop():
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
    t1 = PetriNet.Transition("t1","a")
    t2 = PetriNet.Transition("t2","b")
    t3 = PetriNet.Transition("t3","c")
    t4 = PetriNet.Transition("t4","d")
    t5 = PetriNet.Transition("t5","e")
    t6 = PetriNet.Transition("t6","e")
    t7 = PetriNet.Transition("t7","f")
    t8 = PetriNet.Transition("t8","g")
    model_net.transitions.add(t1)
    model_net.transitions.add(t2)
    model_net.transitions.add(t3)
    model_net.transitions.add(t4)
    model_net.transitions.add(t5)
    model_net.transitions.add(t6)
    model_net.transitions.add(t7)
    model_net.transitions.add(t8)

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

    petri_utils.add_arc_from_to(t7, p7, model_net)
    petri_utils.add_arc_from_to(t8, p7, model_net)

    # add marking for model
    model_im = Marking()
    model_im[p1] = 1
    model_fm = Marking()
    model_fm[p7] = 1
    # gviz = visualizer.apply(model_net, model_im, model_fm)
    # visualizer.view(gviz)
    return model_net, model_im, model_fm


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


def construct_trace_net_without_loop():
    trace_net = PetriNet("trace_net")
    p1 = PetriNet.Place("p1")
    p2 = PetriNet.Place("p2")
    p3 = PetriNet.Place("p3")
    p4 = PetriNet.Place("p4")
    p5 = PetriNet.Place("p5")
    p6 = PetriNet.Place("p6")
    p7 = PetriNet.Place("p7")

    trace_net.places.add(p1)
    trace_net.places.add(p2)
    trace_net.places.add(p3)
    trace_net.places.add(p4)
    trace_net.places.add(p5)
    trace_net.places.add(p6)
    trace_net.places.add(p7)

    # add transitions for trace
    t1 = PetriNet.Transition("t1", "a")
    t2 = PetriNet.Transition("t2", "c")
    t3 = PetriNet.Transition("t3", "e")
    t4 = PetriNet.Transition("t4", "g")
    t5 = PetriNet.Transition("t5", "c")
    t6 = PetriNet.Transition("t6", "d")

    trace_net.transitions.add(t1)
    trace_net.transitions.add(t2)
    trace_net.transitions.add(t3)
    trace_net.transitions.add(t4)
    trace_net.transitions.add(t5)
    trace_net.transitions.add(t6)

    # add arcs for trace
    petri_utils.add_arc_from_to(p1, t1, trace_net)
    petri_utils.add_arc_from_to(p2, t2, trace_net)
    petri_utils.add_arc_from_to(p3, t3, trace_net)
    petri_utils.add_arc_from_to(p4, t4, trace_net)
    petri_utils.add_arc_from_to(p5, t5, trace_net)
    petri_utils.add_arc_from_to(p6, t6, trace_net)

    petri_utils.add_arc_from_to(t1, p2, trace_net)
    petri_utils.add_arc_from_to(t2, p3, trace_net)
    petri_utils.add_arc_from_to(t3, p4, trace_net)
    petri_utils.add_arc_from_to(t4, p5, trace_net)
    petri_utils.add_arc_from_to(t5, p6, trace_net)
    petri_utils.add_arc_from_to(t6, p7, trace_net)

    # add marking for model
    trace_im = Marking()
    trace_im[p1] = 1
    trace_fm = Marking()
    trace_fm[p7] = 1

    # gviz2 = visualizer.apply(trace_net, trace_im, trace_fm)
    # visualizer.view(gviz2)
    return trace_net, trace_im, trace_fm



def construct_trace_net():
    trace_net = PetriNet("trace_net")
    p1 = PetriNet.Place("p1")
    p2 = PetriNet.Place("p2")
    p3 = PetriNet.Place("p3")
    p4 = PetriNet.Place("p4")
    p5 = PetriNet.Place("p5")
    p6 = PetriNet.Place("p6")
    trace_net.places.add(p1)
    trace_net.places.add(p2)
    trace_net.places.add(p3)
    trace_net.places.add(p4)
    trace_net.places.add(p5)
    trace_net.places.add(p6)

    # add transitions for trace
    t1 = PetriNet.Transition("t1", "a")
    t2 = PetriNet.Transition("t2", "d")
    t3 = PetriNet.Transition("t3", "d")
    t4 = PetriNet.Transition("t4", "e")
    t5 = PetriNet.Transition("t5", "g")

    trace_net.transitions.add(t1)
    trace_net.transitions.add(t2)
    trace_net.transitions.add(t3)
    trace_net.transitions.add(t4)
    trace_net.transitions.add(t5)

    # add arcs for trace
    petri_utils.add_arc_from_to(p1, t1, trace_net)
    petri_utils.add_arc_from_to(p2, t2, trace_net)
    petri_utils.add_arc_from_to(p3, t3, trace_net)
    petri_utils.add_arc_from_to(p4, t4, trace_net)
    petri_utils.add_arc_from_to(p5, t5, trace_net)
    petri_utils.add_arc_from_to(t1, p2, trace_net)
    petri_utils.add_arc_from_to(t2, p3, trace_net)
    petri_utils.add_arc_from_to(t3, p4, trace_net)
    petri_utils.add_arc_from_to(t4, p5, trace_net)
    petri_utils.add_arc_from_to(t5, p6, trace_net)

    # add marking for model
    trace_im = Marking()
    trace_im[p1] = 1
    trace_fm = Marking()
    trace_fm[p6] = 1

    # gviz2 = visualizer.apply(trace_net, trace_im, trace_fm)
    # visualizer.view(gviz2)
    return trace_net, trace_im, trace_fm


def reconstruct_alignment(state, visited, queued, traversed, ret_tuple_as_trans_desc=False):
    parent = state.pre_state
    if ret_tuple_as_trans_desc:
        alignment = [(state.pre_transition.name, state.pre_transition.label)]
        while parent.pre_state is not None:
            alignment = [(parent.pre_transition.name, parent.pre_transition.label)] + alignment
            parent = parent.pre_state
    else:
        alignment = [state.pre_transition.label]
        while parent.pre_state is not None:
            alignment = [parent.pre_transition.label] + alignment
            parent = parent.pre_state
    result = {"alignment": alignment, "cost": state.g, "visited_states": visited, "queued": queued,
              "traversed": traversed}
    return result