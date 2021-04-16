from pm4py.objects.petri.importer.variants.pnml import import_net
from astar_implementation import construction, initialization, synchronous_product, astar_v2
from csv import DictWriter
import time
import func_timeout
import pandas as pd
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.conformance.alignments.variants import state_equation_a_star, state_equation_less_memory
from pm4py.objects.log.importer.xes import importer as xes_importer


log_csv = pd.read_csv('F:\State_space_traversal\data\CCC19 - Log CSV.csv', sep=',')
log_csv.rename(columns={'ACTIVITY': 'concept:name'}, inplace=True)
log_csv = dataframe_utils.convert_timestamp_columns_in_df(log_csv)
parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'CASEID'}
event_log = log_converter.apply(log_csv, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)
model_net, model_im, model_fm = import_net('F:\State_space_traversal\data\CCC19 - Model PN.pnml')
field_names = ['alignment', 'cost', 'visited_states', 'queued_states',  'traversed_arcs', 'block_restart','restart',
               'h_recalculation', 'time', 'split', 'lp_solved']

for case_index, case in enumerate(event_log):
    if case_index!=8:
        continue
    trace_lst = []
    for event_index, event in enumerate(event_log[case_index]):
        trace_lst.append(event['concept:name'])

    trace_net, trace_im, trace_fm = construction.construct_trace(trace_lst)
    sync_net, sync_im, sync_fm, sync_index = synchronous_product.construct(trace_net, trace_im, trace_fm, model_net,
                                                                           model_im,
                                                                           model_fm, '>>')
    aux_dict = initialization.initialize_aux_dict(sync_net, sync_im, sync_fm, sync_index)
    print(aux_dict['t_index'])
    start_time = time.time()
    align = astar_v2.astar_with_split(sync_net, sync_im, sync_fm, aux_dict)
    align['time'] = time.time() - start_time
    print(align)
    print("time", time.time() - start_time)
    start_time = time.time()
    align = state_equation_a_star.apply(event_log[case_index], model_net, model_im, model_fm)
    print(align)
    print("time", time.time() - start_time)
    # try:
    #     with open('F:\State_space_traversal\cc_results\ccc_19_astar2.csv', 'a') as f_object:
    #         dictwriter_object = DictWriter(f_object, fieldnames=field_names)
    #         # Pass the dictionary as an argument to the Writerow()
    #         dictwriter_object.writerow(align)
    #         # Close the file object
    #         f_object.close()
    # except func_timeout.exceptions.FunctionTimedOut:
    #     print("timeout", id)
    #     align = {'alignment': "??", 'cost': "??", 'visited_states': "??", 'queued_states': "??", 'traversed_arcs': "??",
    #              'split': "??", 'block_restart': "??", 'h_recalculation': "??", 'time': "??"}
    #     with open('F:\State_space_traversal\cc_results\ccc_19_astar2.csv', 'a') as f_object:
    #         dictwriter_object = DictWriter(f_object, fieldnames=field_names)
    #         # Pass the dictionary as an argument to the Writerow()
    #         dictwriter_object.writerow(align)
    #         # Close the file object
    #         f_object.close()
