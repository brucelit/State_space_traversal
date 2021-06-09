from pm4py.objects.petri.importer.variants.pnml import import_net
import astar
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

import astar_with_check

field_names = ['alignment', 'cost', "visited_states", "queued_states", "traversed_arcs", "lp_solved","restart", "block_restart", "time"]
event_log = xes_importer.apply('C:\data\CCC19 XES.xes')
model_net, model_im, model_fm = import_net('C:\data\CCC19 - Model PN.pnml')
for case_index, case in enumerate(event_log):
    # print(violate_lst)
    start_time = time.time()
    align = astar.apply(case, model_net, model_im, model_fm, {})
    # align['time'] = time.time() - start_time
    # align = state_equation_a_star.apply(case, model_net, model_im, model_fm)
    align['time'] = time.time() - start_time
    # print(align['cost'])
    # print("pm4py", align['cost'])
    # print("time", time.time() - start_time)
    try:
        with open('C:\data\c19_astar_replay_without_add_without_model_rest.csv', 'a') as f_object:
            dictwriter_object = DictWriter(f_object, fieldnames=field_names)
            # Pass the dictionary as an argument to the Writerow()
            dictwriter_object.writerow(align)
            # Close the file object
            f_object.close()
    except func_timeout.exceptions.FunctionTimedOut:
        print("timeout", id)
        align = {'alignment': "??", 'cost': "??", 'visited_states': "??", 'queued_states': "??", 'traversed_arcs': "??",
                 'split': "??", 'block_restart': "??", 'h_recalculation': "??", 'time': "??"}
        with open('C:\data\c19_astar_replay_with_model_rest.csv', 'a') as f_object:
            dictwriter_object = DictWriter(f_object, fieldnames=field_names)
            # Pass the dictionary as an argument to the Writerow()
            dictwriter_object.writerow(align)
            # Close the file object
            f_object.close()
