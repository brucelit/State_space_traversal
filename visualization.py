import tempfile

from graphviz import Digraph

from pm4py.objects.petri.petrinet import Marking
from pm4py.util import exec_utils
from enum import Enum
from pm4py.visualization.petrinet.parameters import Parameters

FORMAT = Parameters.FORMAT
DEBUG = Parameters.DEBUG
RANKDIR = Parameters.RANKDIR


def graphviz_visualization(net, image_format="png", initial_marking=None, final_marking=None, current_marking=None):
    if initial_marking is None:
        initial_marking = Marking()
    if final_marking is None:
        final_marking = Marking()
    if current_marking is None:
        current_marking = Marking()
    filename = tempfile.NamedTemporaryFile(suffix='.gv')
    viz = Digraph(name='father', filename=filename.name, engine='dot', graph_attr={'label': 'Title',
                                                                                   'labelloc': 't',
                                                                                   'fontsize': '25',
                                                                                   'bgcolor': 'transparent',
                                                                                   'rankdir': 'TB', 'ranksep': '2'})
    viz_trace_trans = Digraph(name='child1', engine='dot', graph_attr={'bgcolor': 'transparent', 'rank': 'source'})
    viz_sync_trans = Digraph(name='child2', engine='dot', graph_attr={'bgcolor': 'transparent', 'rank': 'same'})
    viz_model = Digraph(name='father2', engine='dot', graph_attr={'bgcolor': 'transparent'})
    viz_model_place = Digraph(name='child4', engine='dot', graph_attr={'bgcolor': 'transparent', 'rank': 'same'})
    viz_model_trans = Digraph(name='child3', engine='dot', graph_attr={'bgcolor': 'transparent', 'rank': 'sink'})
    # transitions
    viz.attr('node', shape='square', width='0.9')
    viz_trace_trans.attr('node', shape='square', width='0.9')
    viz_sync_trans.attr('node', shape='square', width='0.9')
    viz_model.attr('node', shape='square', width='0.9')
    viz_model_trans.attr('node', shape='square', width='0.9')
    # add transitions, in order by their (unique) name, to avoid undeterminism in the visualization
    trans_sort_list = sorted(list(net.transitions), key=lambda x: (x.label if x.label is not None else "tau", x.name))
    for t in trans_sort_list:
        if t.label is not None:
            if t.label[0] == ">>":
                viz_model_trans.node(str(id(t)), str(t.label), style='filled', fillcolor="#FFFFFF")
            elif t.label[1] == ">>":
                viz_trace_trans.node(str(id(t)), str(t.label), style='filled', fillcolor="#C0C0C0")
            else:
                viz_sync_trans.node(str(id(t)), str(t.label), style='filled', fillcolor="#F5F5F5")
    viz_model.subgraph(viz_model_trans)
    viz_model.subgraph(viz_model_place)
    viz.subgraph(viz_trace_trans)
    viz.subgraph(viz_sync_trans)
    viz.subgraph(viz_model)
    # places
    viz.attr('node', shape='circle', fixedsize='true', width='1')
    viz_model.attr('node', shape='circle', fixedsize='true', width='1')
    viz_trace_trans.attr('node', shape='circle', fixedsize='true', width='1')
    viz_model_place.attr('node', shape='circle', fixedsize='true', width='1')
    viz_model_trans.attr('node', shape='circle', fixedsize='true', width='1')
    # add places, in order by their (unique) name, to avoid undeterminism in the visualization
    places_sort_list_im = sorted([x for x in list(net.places) if x in initial_marking], key=lambda x: x.name)
    places_sort_list_fm = sorted([x for x in list(net.places) if x in final_marking and not x in initial_marking],
                                 key=lambda x: x.name)
    places_sort_list_not_im_fm = sorted(
        [x for x in list(net.places) if x not in initial_marking and x not in final_marking], key=lambda x: x.name)
    places_sort_list = places_sort_list_im + places_sort_list_not_im_fm + places_sort_list_fm

    for p in places_sort_list:
        if p.name[1] == ">>":
            viz_trace_trans.node(str(id(p)), str(p.name), fontsize='13')
        elif p.name[0] == ">>" and (p in initial_marking or p in final_marking):
            viz_model_trans.node(str(id(p)), str(p.name), fontsize='13')
        else:
            viz_model_place.node(str(id(p)), str(p.name), fontsize='13')
    viz_model.subgraph(viz_model_place)
    viz_model.subgraph(viz_model_trans)
    viz.subgraph(viz_model)
    viz.subgraph(viz_trace_trans)
    # add arcs, in order by their source and target objects names, to avoid undeterminism in the visualization
    arcs_sort_list = sorted(list(net.arcs), key=lambda x: (x.source.name, x.target.name))
    for a in arcs_sort_list:
        viz.edge(str(id(a.source)), str(id(a.target)))
    viz.attr(overlap='false')
    # viz.attr(fontsize='11')
    viz.format = image_format
    return viz, places_sort_list


def graphviz_state_change(viz, places_sort_list, current_marking):
    for p in places_sort_list:
        if p in current_marking:
            viz.node(str(id(p)), str(p.name), style='filled', fillcolor="#BDFCC9", fontsize='13')
        else:
            viz.node(str(id(p)), str(p.name), style='filled', fillcolor="#FFFFFF", fontsize='13')
    return viz
