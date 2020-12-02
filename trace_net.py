import numpy as np
from scipy.optimize import linprog
from pm4py.objects import petri
from pm4py.objects.petri import utils as petri_utils
from pm4py.objects.petri.petrinet import PetriNet, Marking
import pulp
from pm4py.visualization.petrinet import visualizer


def construct_trace_net_without_loop():
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
    t2 = PetriNet.Transition("t2", "c")
    t3 = PetriNet.Transition("t3", "d")
    t4 = PetriNet.Transition("t4", "g")
    t5 = PetriNet.Transition("t5", "e")

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

    gviz2 = visualizer.apply(trace_net, trace_im, trace_fm)
    visualizer.view(gviz2)
    return trace_net, trace_im, trace_fm
