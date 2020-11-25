import tempfile
import tempfile
from copy import copy
import os

from graphviz import Digraph
from pm4py.util import exec_utils
from pm4py.visualization.transition_system.parameters import Parameters
from graphviz import Digraph
from copy import copy

from pm4py.objects.petri.petrinet import Marking
from pm4py.util import exec_utils
from enum import Enum
from pm4py.visualization.petrinet.parameters import Parameters
from pm4py.visualization.transition_system import visualizer as ts_visualizer


def visualize_transition_system(ts):
    for state in ts.states:
        state.label = state.name
    filename = tempfile.NamedTemporaryFile(suffix='.gv')
    viz = Digraph(ts.name, filename=filename.name, engine='dot', graph_attr={'bgcolor': 'transparent'})
    viz.attr(overlap='false')
    viz.attr(fontsize='11')
    viz.format = "png"
    return viz


def viz_state_change(ts, curr_state, valide_state_lst, invalide_state_lst):
    for state in ts.states:
        state.label = state.name
    filename = tempfile.NamedTemporaryFile(suffix='.gv')
    viz = Digraph(ts.name, filename=filename.name, engine='dot', graph_attr={'bgcolor': 'transparent'})
    viz.attr('node', shape='box', fixedsize='true', width='1', fontsize = '13')
    for s in ts.states:
        if s.name in curr_state:
            viz.node(str(id(s)), str(s.label), style='filled', fillcolor="#1E90FF")
        elif s.name in valide_state_lst:
            viz.node(str(id(s)), str(s.label), style='filled', fillcolor="#228B22")
        elif s.name in invalide_state_lst:
            viz.node(str(id(s)), str(s.label), style='filled', fillcolor="#FF0000")
        else:
            pass
    for t in ts.transitions:
        if t.to_state.name in invalide_state_lst and (t.from_state.name in valide_state_lst or t.from_state.name in curr_state):
            viz.edge(str(id(t.from_state)), str(id(t.to_state)), label=t.name)
        elif t.to_state.name in valide_state_lst and (t.from_state.name in valide_state_lst or t.from_state.name in curr_state):
            viz.edge(str(id(t.from_state)), str(id(t.to_state)), label=t.name)
        else:
            pass
    viz.attr(overlap='false')
    viz.attr(fontsize='13')
    viz.format = "png"
    # ts_visualizer.save(viz, os.path.join("E:/Thesis/img", "step" + str(order) + ".png"))
    return viz
