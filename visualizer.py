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
    """
    Provides visualization for the petrinet
    Parameters
    ----------
    net: :class:`pm4py.entities.petri.petrinet.PetriNet`
        Petri net
    image_format
        Format that should be associated to the image
    initial_marking
        Initial marking of the Petri net
    final_marking
        Final marking of the Petri net
    decorations
        Decorations of the Petri net (says how element must be presented)
    debug
        Enables debug mode
    set_rankdir
        Sets the rankdir to LR (horizontal layout)
    Returns
    -------
    viz :
        Returns a graph object
    """
    if initial_marking is None:
        initial_marking = Marking()
    if final_marking is None:
        final_marking = Marking()
    if current_marking is None:
        current_marking = Marking()
    filename = tempfile.NamedTemporaryFile(suffix='.gv')
    viz = Digraph(net.name, filename=filename.name, engine='dot',
                  graph_attr={'bgcolor': 'transparent', 'rankdir': 'TB'})
    viz2 = Digraph(name='child1', engine='dot', graph_attr={'bgcolor': 'transparent', 'rank': 'max'})
    viz3 = Digraph(name='child2', engine='dot', graph_attr={'bgcolor': 'transparent', 'rank': 'min'})
    viz4 = Digraph(name='child3', engine='dot', graph_attr={'bgcolor': 'transparent', 'rank': 'same'})
    # transitions of
    viz.attr('node', shape='square', width='0.9')
    # add transitions, in order by their (unique) name, to avoid undeterminism in the visualization
    trans_sort_list = sorted(list(net.transitions), key=lambda x: (x.label if x.label is not None else "tau", x.name))
    for t in trans_sort_list:
        if t.label is not None:
            if t.label[0] == ">>":
                viz2.node(str(id(t)), str(t.label), style='filled', fillcolor="#FFFFFF")
            elif t.label[1] == ">>":
                viz3.node(str(id(t)), str(t.label), style='filled', fillcolor="#C0C0C0")
            else:
                viz4.node(str(id(t)), str(t.label), style='filled', fillcolor="#F5F5F5")
    viz.subgraph(viz2)
    viz.subgraph(viz3)
    viz.subgraph(viz4)
    # places
    viz.attr('node', shape='circle', fixedsize='true', width='0.8')
    # add places, in order by their (unique) name, to avoid undeterminism in the visualization
    places_sort_list_im = sorted([x for x in list(net.places) if x in initial_marking], key=lambda x: x.name)
    places_sort_list_fm = sorted([x for x in list(net.places) if x in final_marking and not x in initial_marking],
                                 key=lambda x: x.name)
    places_sort_list_not_im_fm = sorted(
        [x for x in list(net.places) if x not in initial_marking and x not in final_marking], key=lambda x: x.name)
    # making the addition happen in this order:
    # - first, the places belonging to the initial marking
    # - after, the places not belonging neither to the initial marking and the final marking
    # - at last, the places belonging to the final marking (but not to the initial marking)
    # in this way, is more probable that the initial marking is on the left and the final on the right
    places_sort_list = places_sort_list_im + places_sort_list_not_im_fm + places_sort_list_fm

    for p in places_sort_list:
        if p in current_marking:
            viz.node(str(id(p)), str(p.name), style='filled', fillcolor="#BDFCC9", fontsize='13')
        else:
            viz.node(str(id(p)), str(p.name), fontsize='13')

    # add arcs, in order by their source and target objects names, to avoid undeterminism in the visualization
    arcs_sort_list = sorted(list(net.arcs), key=lambda x: (x.source.name, x.target.name))
    for a in arcs_sort_list:
        viz.edge(str(id(a.source)), str(id(a.target)))
    viz.attr(overlap='false')
    viz.attr(fontsize='11')
    viz.format = image_format
    return viz, places_sort_list


def graphviz_state_change(viz, places_sort_list, current_marking):
    for p in places_sort_list:
        if p in current_marking:
            viz.node(str(id(p)), str(p.name), style='filled', fillcolor="#BDFCC9", fontsize='13')
        else:
            viz.node(str(id(p)), str(p.name),  style='filled', fillcolor="#FFFFFF", fontsize='13')
    return viz