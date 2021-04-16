import time
from tqdm import tqdm
from csv import DictWriter
import func_timeout
from pm4py.algo.conformance.alignments.variants import state_equation_a_star, state_equation_less_memory
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri.importer.variants.pnml import import_net
from astar_implementation import construction, initialization, synchronous_product, astar_v2, astar, astar2
from pm4py.objects.petri.incidence_matrix import construct as inc_mat_construct

field_names = ['alignment', 'cost', 'queued_states', 'visited_states', 'traversed_arcs', 'h_recalculation', 'split', 'block_restart','restart', 'time']

# field_names = ['alignment', 'cost', "visited_states", "queued_states", "traversed_arcs", "lp_solved","time"]
event_log = xes_importer.apply('F:\State_space_traversal\data\log_b_178.xes')
model_net, model_im, model_fm = import_net('F:\State_space_traversal\data\model_b.pnml')
for case_index, case in enumerate(event_log):
    # if case_index > 617:
    #     continue
    # trace_lst = []
    # for event_index, event in enumerate(event_log[case_index]):
    #     trace_lst.append(event['concept:name'])
    # trace_net, trace_im, trace_fm, trace_name_lst = construction.construct_trace(trace_lst)
    # sync_net, sync_im, sync_fm, sync_index = synchronous_product.construct(trace_net, trace_im, trace_fm, model_net,
    #                                                                        model_im, model_fm, '>>')
    # aux_dict = initialization.initialize_aux_dict(sync_net, sync_im, sync_fm, sync_index)
    # align = astar_v2.astar_with_split(sync_net, sync_im, sync_fm, aux_dict)
    # print(align)
    # align = state_equation_a_star.apply(case, model_net, model_im, model_fm)
    # print(align)
    # align2 = astar.apply(case, model_net, model_im, model_fm)
    # print(align2)
    # if align['cost']*10000 != align2['cost']:
    #     print('case index', case_index)

    # start rule checking
    # print(trace_lst)
    # ic = inc_mat_construct(model_net)
    # dict_l = ic.dict_l
    # dict_r = ic.dict_r
    # i = 1
    # rule_lst = [i for i in range(0, len(dict_r))]
    #
    # violate_lst = {}
    # while i < len(trace_lst):
    #     rule1 = trace_lst[0:i]
    #     rule2 = set(rule1)
    #     for j in rule_lst:
    #         count_l = len(rule2.intersection(set(dict_l[j])))
    #         count_2 = len(rule2.intersection(set(dict_r[j])))
    #         if count_l < count_2:
    #             rule_lst.remove(j)
    #             violate_lst[rule1[-1]] = i
    #             continue
    #     i += 1
    # print("violate_lst", violate_lst)

    try:
        start_time = time.time()
        align = state_equation_a_star.apply(case, model_net, model_im, model_fm)
        # align['time'] = time.time() - start_time
        # print(align)
        # align = astar_v2.astar_with_split(sync_net, sync_im, sync_fm, aux_dict)
        print(align)
        # align = astar_v2.apply(case, model_net, model_im, model_fm)
        # align['time'] = time.time() - start_time
        # print(align)
        # with open('F:\State_space_traversal\cc_results\c20b_astar.csv', 'a') as f_object:
        #     dictwriter_object = DictWriter(f_object, fieldnames=field_names)
        #     # Pass the dictionary as an argument to the Writerow()
        #     dictwriter_object.writerow(align)
        #     # Close the file object
        #     f_object.close()
    except func_timeout.exceptions.FunctionTimedOut:
        print("timeout", id)
        align = {'alignment': "??", 'cost': "??"}
        with open('F:\State_space_traversal\cc_results\c20b_astar.csv', 'a') as f_object:
            dictwriter_object = DictWriter(f_object, fieldnames=field_names)
            # Pass the dictionary as an argument to the Writerow()
            dictwriter_object.writerow(align)
            # Close the file object
            f_object.close()