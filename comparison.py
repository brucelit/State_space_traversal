from pm4py.algo.conformance.alignments.variants import state_equation_a_star, state_equation_less_memory
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri.importer.variants.pnml import import_net
import time
from astar_implementation import construction
from pm4py.visualization.petrinet import visualizer
from astar_implementation import construction, astar_latest, initialization, synchronous_product
from csv import DictWriter
from func_timeout import func_set_timeout
import time
import func_timeout

log = xes_importer.apply('E:\Thesis\ccc2019\CCC19 - Log CSV.csv')
model_net, model_im, model_fm = import_net('E:\Thesis\ccc2019\CCC19 - Model PN.pnml')
# gviz = visualizer.apply(model_net, model_im, model_fm)
# visualizer.view(gviz)
field_names = ['alignment', 'cost', 'visited_states', 'queued_states', 'traversed_arcs', 'split',
               'block_restart', 'h_recalculation', 'time']
# trace_lst_lst = []
for case_index, case in enumerate(log):
    trace_lst = []
    for event_index, event in enumerate(log[case_index]):
        trace_lst.append(event["concept:name"])

    # if trace_lst in trace_lst_lst:
    #     continue
    # else:
    #     trace_lst_lst.append(trace_lst)
    # trace_net, trace_im, trace_fm = construction.construct_trace(trace_lst)
    # p1 = state_equation_a_star.Parameters
    # start_time = time.time()
    # dict1 = state_equation_a_star.apply(log[case_index], model_net, model_im, model_fm)



    trace_net, trace_im, trace_fm = construction.construct_trace(trace_lst)
    sync_net, sync_im, sync_fm, sync_index = synchronous_product.construct(trace_net, trace_im, trace_fm, model_net,
                                                                           model_im,
                                                                           model_fm, '>>')
    aux_dict = initialization.initialize_aux_dict(sync_net, sync_im, sync_fm, sync_index)
    start_time = time.time()
    # print(aux_dict['cost_vec'])
    # print(aux_dict['t_index'])
    # print("inci matrix", aux_dict['incidence_matrix'])
    # print("consumption matrix", aux_dict['consumption_matrix'])
    # print("ini vec", aux_dict['ini_vec'])
    try:
        align = astar_latest.astar_with_split(sync_net, sync_im, sync_fm, aux_dict)
        align['time'] = time.time() - start_time
        print(align)
        with open('ccc_19.csv', 'a') as f_object:
            dictwriter_object = DictWriter(f_object, fieldnames=field_names)
            # Pass the dictionary as an argument to the Writerow()
            dictwriter_object.writerow(align)
            # Close the file object
            f_object.close()
    except func_timeout.exceptions.FunctionTimedOut:
        print("timeout", id)
        align = {'alignment': "??", 'cost': "??", 'visited_states': "??", 'queued_states': "??", 'traversed_arcs': "??",
                 'split': "??", 'block_restart': "??", 'h_recalculation': "??", 'time': "??"}
        with open('ccc_19.csv', 'a') as f_object:
            dictwriter_object = DictWriter(f_object, fieldnames=field_names)
            # Pass the dictionary as an argument to the Writerow()
            dictwriter_object.writerow(align)
            # Close the file object
            f_object.close()

# sync_net, sync_im, sync_fm, sync_index = synchronous_product.construct(trace_net, trace_im, trace_fm, model_net,
#                                                                        model_im,
#                                                                        model_fm, '>>')
# aux_dict = initialization.initialize_aux_dict(sync_net, sync_im, sync_fm, sync_index)
# split_lst = {None: -1}
# start_time = time.time()
# # print(aux_dict['cost_vec'])
# # print(aux_dict['t_index'])
# print("", aux_dict['incidence_matrix'])
# align = astar_latest.astar_with_split(sync_net, sync_im, sync_fm, aux_dict, split_lst)
# print(align)
# print("--- %s seconds ---" % (time.time() - start_time))
# # field_names = ['trace', 'cost', 'visited_states', 'queued_states', 'traversed_arcs']
# with open('comparison2.csv', 'a') as f_object:
#     dictwriter_object = DictWriter(f_object, fieldnames=field_names)
#
#     # Pass the dictionary as an argument to the Writerow()
#     dictwriter_object.writerow(align)
#
#     # Close the file object
#     f_object.close()
