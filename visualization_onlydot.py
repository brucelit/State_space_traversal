import tempfile

from graphviz import Digraph


def visualize_transition_system(ts):
    for state in ts.states:
        state.label = state.name
    filename = tempfile.NamedTemporaryFile(suffix='.gv')
    viz = Digraph(ts.name, filename=filename.name, engine='dot', graph_attr={'bgcolor': 'transparent'})
    viz.attr(overlap='false')
    viz.attr(fontsize='11')
    viz.format = "png"
    return viz


def gviz_state_change(ts, curr_state, valide_state_lst, invalide_state_lst, visited, split_lst, open_set):
    for state in ts.states:
        state.label = state.name
    filename = tempfile.NamedTemporaryFile(suffix='.gv')
    viz = Digraph(ts.name, filename=filename.name, engine='dot', graph_attr={'bgcolor': 'transparent'})
    viz.attr('node', shape='circle', fixedsize='true', width='0.4')
    for s in ts.states:
        if s.name in curr_state:
            viz.node(str(id(s)), str(""), style='filled', fillcolor="#FFFFFF")
        elif s.name in valide_state_lst:
            viz.node(str(id(s)), str(""), style='filled', fillcolor="#1E90FF")
        elif s.name in invalide_state_lst:
            viz.node(str(id(s)), str(""), style='filled', fillcolor="#FF0000")
        else:
            pass
    for t in ts.transitions:
        if t.from_state.name in invalide_state_lst and (t.to_state.name in valide_state_lst or t.to_state.name in invalide_state_lst):
            viz.edge(str(id(t.from_state)), str(id(t.to_state)), label="")
        elif t.from_state.name in valide_state_lst and (t.to_state.name in valide_state_lst or t.to_state.name in invalide_state_lst):
            viz.edge(str(id(t.from_state)), str(id(t.to_state)), label="")
        elif t.from_state.name in curr_state and (t.to_state.name in valide_state_lst or t.to_state.name in invalide_state_lst):
            viz.edge(str(id(t.from_state)), str(id(t.to_state)), label="")
        else:
            pass
    viz.attr(overlap='false')
    viz.attr(fontsize='8')
    viz.format = "png"
    viz.graph_attr['label'] = "\nNumber of states visited: " + str(visited) + "\nsplit list:" + \
                              str(list(split_lst.keys())[1:]) + "\nNumber of states in open set: " + str(len(open_set))
    return viz



def viz_state_change(ts, curr_state, valide_state_lst, invalide_state_lst, visited, split_lst, open_set):
    for state in ts.states:
        state.label = state.name
    filename = tempfile.NamedTemporaryFile(suffix='.gv')
    viz = Digraph(ts.name, filename=filename.name, engine='dot', graph_attr={'bgcolor': 'transparent'})
    viz.attr('node', shape='box', fixedsize='true', width='1', fontsize = '13')
    for s in ts.states:
        if s.name in curr_state:
            viz.node(str(id(s)), str(s.label), style='filled', fillcolor="#FFFFFF")
        elif s.name in valide_state_lst:
            viz.node(str(id(s)), str(s.label), style='filled', fillcolor="#1E90FF")
        elif s.name in invalide_state_lst:
            viz.node(str(id(s)), str(s.label), style='filled', fillcolor="#FF0000")
        else:
            pass
    for t in ts.transitions:
        if t.from_state.name in invalide_state_lst and (t.to_state.name in valide_state_lst or t.to_state.name in invalide_state_lst):
            viz.edge(str(id(t.from_state)), str(id(t.to_state)), label=t.name)
        elif t.from_state.name in valide_state_lst and (t.to_state.name in valide_state_lst or t.to_state.name in invalide_state_lst):
            viz.edge(str(id(t.from_state)), str(id(t.to_state)), label=t.name)
        elif t.from_state.name in curr_state and (t.to_state.name in valide_state_lst or t.to_state.name in invalide_state_lst):
            viz.edge(str(id(t.from_state)), str(id(t.to_state)), label=t.name)
        else:
            pass
    viz.attr(overlap='false')
    viz.attr(fontsize='13')
    viz.format = "png"
    viz.graph_attr['label'] = "\nNumber of states visited: " + str(visited) + "\nsplit list:" + \
                              str(split_lst.keys()) + "\nNumber of states in open set: " + str(len(open_set))
    return viz
