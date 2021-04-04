import time
from csv import DictWriter
import func_timeout
from pm4py.algo.conformance.alignments.variants import state_equation_a_star, state_equation_less_memory
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri.importer.variants.pnml import import_net
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.petri.importer.variants.pnml import import_net
from astar_implementation import construction, initialization, synchronous_product, astar_v2

field_names = ['alignment', 'cost', 'queued_states', 'visited_states', 'traversed_arcs', 'h_recalculation', 'split','block_restart', 'time']
event_log = xes_importer.apply('F:\State_space_traversal\data\log_b_178.xes')
model_net, model_im, model_fm = import_net('F:\State_space_traversal\data\model_b.pnml')
for case_index, case in enumerate(event_log):
    # print(case)
    trace_lst = []
    for event_index, event in enumerate(event_log[case_index]):
        trace_lst.append(event['concept:name'])
    trace_net, trace_im, trace_fm = construction.construct_trace(trace_lst)
    sync_net, sync_im, sync_fm, sync_index = synchronous_product.construct(trace_net, trace_im, trace_fm, model_net,
                                                                           model_im,
                                                                           model_fm, '>>')
    aux_dict = initialization.initialize_aux_dict(sync_net, sync_im, sync_fm, sync_index)
    try:
        start_time = time.time()
        align = astar_v2.astar_with_split(sync_net, sync_im, sync_fm, aux_dict)
        align['time'] = time.time() - start_time
        print(align)
        # with open('F:\State_space_traversal\cc_results\c20b_astar_latest.csv', 'a') as f_object:
        #     dictwriter_object = DictWriter(f_object, fieldnames=field_names)
        #     # Pass the dictionary as an argument to the Writerow()
        #     dictwriter_object.writerow(align)
        #     # Close the file object
        #     f_object.close()
    except func_timeout.exceptions.FunctionTimedOut:
        print("timeout", id)
        align = {'alignment': "??", 'cost': "??", 'visited_states': "??", 'queued_states': "??", 'traversed_arcs': "??",
                'h_recalculation': "??",'split':"??",'block_restart':"??",'time':'??'}
        with open('F:\State_space_traversal\cc_results\c20b_astar_latest.csv', 'a') as f_object:
            dictwriter_object = DictWriter(f_object, fieldnames=field_names)
            # Pass the dictionary as an argument to the Writerow()
            dictwriter_object.writerow(align)
            # Close the file object
            f_object.close()