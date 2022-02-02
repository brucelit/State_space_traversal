import func_timeout
import statistics
from csv import DictWriter
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer.variants.pnml import import_net
from pm4py.algo.conformance.alignments.petri_net.variants import state_equation_a_star
import warnings
import pandas as pd

import astar_bid
import astar_cache_pp_old
import astar_pm4py
import astar_precompute
import astar_reverse
import astar_tue
from tqdm import tqdm
import astar_tue_pp
import astar_tue_test
import construction


def search():
    # Here to change the log file in dataset: the .xes file
    event_log = xes_importer.apply('data\Log_BPIC19_2_new.xes')

    # event_log = xes_importer.apply('data\Log_BPIC19_2_7.xes')
    # Here to change the model in dataset: the .pnml file
    model_net, model_im, model_fm = import_net('F:\Thesis\data\BPIC19_2_IM.pnml')

    # the colunm name in result csv filez
    field_names = ["time_sum",
                   "time_heuristic",
                   "time_heap",
                   "lp_solved",
                   "lp_for_ini_solved",
                   "restart",
                   "visited_states",
                   "queued_states",
                   "traversed_arcs",
                   "num_insert",
                   "num_pop",
                   "num_update",
                   "num_retrieval",
                   "split_num",
                   "trace_length",
                   "cost"]

    df = pd.DataFrame(columns=field_names)
    df.to_csv('F:\Thesis\data\BPIC19_2\BPIC_tue_1822_0129.csv', sep=',', index=False)

    # iterate every case in this xes log file
    for case_index in tqdm(range(len(event_log))):
        result = {'time_sum': 0, 'time_heuristic': 0, 'time_heap': 0, 'split_num': 0,
                  'restart': 0, 'visited_states': 0, "queued_states": 0, 'traversed_arcs': 0,
                  'lp_solved': 0, 'lp_for_ini_solved': 0,
                  'num_insert': 0, 'num_pop': 0, 'num_retrieval': 0, 'num_update': 0,
                  'trace_length': 0, 'cost': 0}
        if case_index <= 1800:
            continue
        # if case_index > 1800:
        #     break
        '''
        # Choose one of the following align, then save the results in csv file for further analysis
        # Choice 1: the original algorithm in paper "Efficiently computing alignments algorithm
        # and datastructures" from Eindhoven University
        '''
        # Choice 8: the algorithm from pm4py
        # align = astar_pm4py.apply(event_log[case_index], model_net, model_im, model_fm)

        # align = state_equation_a_star.apply(event_log[case_index], model_net, model_im, model_fm)
        # align['time_heap'] = 0
        # align['restart'] = 0
        # align['trace_length'] = 0

        # align = astar_tue_test.apply(event_log[case_index], model_net, model_im, model_fm)

        align1 = astar_tue.Inc_astar(event_log[case_index], model_net, model_im, model_fm)
        align = align1.apply(event_log[case_index], model_net, model_im, model_fm)
        print(align)

        # align1 = astar_tue_test.Inc_astar(event_log[case_index], model_net, model_im, model_fm)
        # align = align1.apply(event_log[case_index], model_net, model_im, model_fm)
        # print(align)

        # align1 = astar_bid.Inc_astar(event_log[case_index], model_net, model_im, model_fm)
        # align = align1.apply(event_log[case_index], model_net, model_im, model_fm)

        # align1 = astar_tue_pp.Inc_astar(event_log[case_index], model_net, model_im, model_fm)
        # align = align1.apply(event_log[case_index], model_net, model_im, model_fm)
        # print(align)

        # align1 = astar_reverse.Inc_astar(event_log[case_index], model_net, model_im, model_fm)
        # align = align1.apply(event_log[case_index], model_net, model_im, model_fm)
        # print(align)
        # print(align['cost'], align["time_sum"], align['lp_solved'], align['visited_states'])

        # Choice 2: the algorithm from in paper "Improving Alignment Computation
        # using Model-based Preprocessing" from Eindhoven University
        # trace_lst = 0
        # for event_index, event in enumerate(event_log[case_index]):
        #     trace_lst = event['concept:name'])
        # ic = astar_precompute.construct(model_net)
        # split_list = construction.precompute_forward(trace_lst, ic)
        # align1 = astar_precompute.Inc_astar(event_log[case_index], model_net, model_im, model_fm)
        # align = align1.apply(event_log[case_index], model_net, model_im, model_fm, split_list)
        '''
        # Choice 3: the algorithm from litian
        align = astar_bid.apply(case, model_net, model_im, model_fm)
        '''
        # align = astar_bid.apply(case, model_net, model_im, model_fm)

        # Choice 8: the algorithm from pm4py
        align = state_equation_a_star.apply(event_log[case_index], model_net, model_im, model_fm)
        # print(align)
        # align = astar_pm4py.apply(case, model_net, model_im, model_fm)
        # print(align)
        result['time_sum'] = align['time_sum']
        result['time_heuristic'] = align['time_heuristic']
        result['time_heap'] = align['time_heap']
        result['visited_states'] = align['visited_states']
        result['queued_states'] = align['queued_states']
        result['traversed_arcs'] = align['traversed_arcs']
        result['lp_solved'] = align['lp_solved']
        result['lp_for_ini_solved'] = align['lp_for_ini_solved']
        result['num_insert'] = align['num_insert']
        result['num_pop'] = align['num_pop']
        result['num_retrieval'] = align['num_retrieval']
        result['num_update'] = align['num_update']
        result['restart'] = align['restart']
        result['split_num'] = align['split_num']
        result['trace_length'] = align['trace_length']
        result['cost'] = align['cost']

        with open('F:\Thesis\data\BPIC19_2\BPIC_tue_1822_0129.csv', 'a') as f_object:
            dictwriter_object = DictWriter(f_object, fieldnames=field_names)
            # Pass the dictionary as an argument to the Writerow()
            dictwriter_object.writerow(result)
            # Close the file object
            f_object.close()

    df = pd.read_csv('F:\Thesis\data\BPIC19_2\BPIC_tue_1822_0129.csv')
    total = df.sum()
    df2 = pd.DataFrame([total.transpose()], columns=["time_sum",
                                                     "time_heuristic",
                                                     "time_heap",
                                                     "lp_solved",
                                                     "lp_for_ini_solved",
                                                     "restart",
                                                     "visited_states",
                                                     "queued_states",
                                                     "traversed_arcs",
                                                     "num_insert",
                                                     "num_pop",
                                                     "num_update",
                                                     "num_retrieval",
                                                     "split_num",
                                                     "trace_length",
                                                     "cost"])
    df3 = pd.concat([df2, df]).reset_index(drop=True)
    df3.to_csv('F:\Thesis\data\BPIC19_2\BPIC_tue_1822_0129.csv', index=False)


if __name__ == "__main__":
    search()
