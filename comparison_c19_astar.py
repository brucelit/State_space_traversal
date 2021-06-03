from pm4py.objects.petri.importer.variants.pnml import import_net
import construction, initialization, synchronous_product, astar, astar_check
from csv import DictWriter
import time
import func_timeout
import pandas as pd
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.conformance.alignments.variants import state_equation_a_star, state_equation_less_memory
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri.incidence_matrix import construct as inc_mat_construct
from collections import Counter


field_names = ['alignment', 'cost', "visited_states", "queued_states", "traversed_arcs", "lp_solved","restart", "block_restart", "time"]
event_log = xes_importer.apply('C:\data\CCC19 XES.xes')
model_net, model_im, model_fm = import_net('C:\data\CCC19 - Model PN.pnml')
for case_index, case in enumerate(event_log):
    trace_lst = []
    for event_index, event in enumerate(event_log[case_index]):
        trace_lst.append(event['concept:name'])
    trace_net, trace_im, trace_fm, trace_name_lst = construction.construct_trace(trace_lst)
    ic = inc_mat_construct(model_net)
    dict_l = ic.rule_l
    dict_r = ic.rule_r
    i = 1
    rule_lst = [i for i in range(0, len(dict_r))]
    violate_lst = {}
    while i < len(trace_lst):
        trace_prefix = Counter(trace_lst[0:i])
        # print(trace_prefix)
        for j in rule_lst:
            count_l = 0
            count_r = 0
            for k, v in trace_prefix.items():
                if k in dict_l[j]:
                    count_l += v
                if k in dict_r[j]:
                    count_r += v
            if count_l < count_r:
                # print(dict_l[j], dict_r[j], count_l, count_r, trace_lst[0:i])
                rule_lst.remove(j)
                # print(len(rule_lst))
                violate_lst[trace_lst[0:i][-1]] = i
                continue
        i += 1
        # print("vl", violate_lst)
    lst = list(violate_lst.values())
    lst2 = []
    for i in range(len(lst)):
        if i+1 < len(lst) and lst[i] + 1 == lst[i+1]:
            lst2.append(lst[i])
        i += 1

    v2 = {key:val for key, val in violate_lst.items() if val not in lst2}
    # print(violate_lst)
    start_time = time.time()
    align = astar.apply(case, model_net, model_im, model_fm, {}, trace_lst)
    align['time'] = time.time() - start_time
    # align = state_equation_a_star.apply(case, model_net, model_im, model_fm)
    # align['time'] = time.time() - start_time

    # print("pm4py", align['cost'])
    # print("time", time.time() - start_time)
    try:
        with open('C:\data\c19_astar_noreplay.csv', 'a') as f_object:
            dictwriter_object = DictWriter(f_object, fieldnames=field_names)
            # Pass the dictionary as an argument to the Writerow()
            dictwriter_object.writerow(align)
            # Close the file object
            f_object.close()
    except func_timeout.exceptions.FunctionTimedOut:
        print("timeout", id)
        align = {'alignment': "??", 'cost': "??", 'visited_states': "??", 'queued_states': "??", 'traversed_arcs': "??",
                 'split': "??", 'block_restart': "??", 'h_recalculation': "??", 'time': "??"}
        with open('C:\data\c19_astar_noreplay.csv', 'a') as f_object:
            dictwriter_object = DictWriter(f_object, fieldnames=field_names)
            # Pass the dictionary as an argument to the Writerow()
            dictwriter_object.writerow(align)
            # Close the file object
            f_object.close()
