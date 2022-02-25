import func_timeout
import statistics
from csv import DictWriter
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer.variants.pnml import import_net
from pm4py.algo.conformance.alignments.petri_net.variants import state_equation_a_star
import warnings
import pandas as pd

import astar_bid
import astar_pm4py
import astar_precompute
import astar_reverse
import astar_tue
from tqdm import tqdm

import astar_tue_open_set_heapq
import astar_tue_open_set_hotq
import astar_tue_pp
import construction


def search():
    # Here to change the log file in dataset: the .xes file
    event_log = xes_importer.apply('log\Sepsisn0.xes')
    model_net, model_im, model_fm = import_net('model\sepsis02.pnml')

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
                   "reopen_close",
                   "heap_insert",
                   "heap_remove",
                   "heap_update",
                   "heap_retrieval",
                   "heap_total",
                   "split_num",
                   "trace_length",
                   "alignment_length",
                   "cost"]
    align_lst = ["", ]

    df = pd.DataFrame(columns=field_names)
    df.to_csv('F:\Thesis\data\data_sepsis\sepsis_tue_open_set_heapq_original.csv', sep=',', index=False)

    # iterate every case in this xes log file
    for case_index in tqdm(range(len(event_log))):
        result = {'time_sum': 0, 'time_heuristic': 0, 'time_heap': 0, 'split_num': 0,
                  'restart': 0, 'visited_states': 0, "queued_states": 0, 'traversed_arcs': 0,
                  'lp_solved': 0, 'lp_for_ini_solved': 0, "reopen_close": 0,
                  'heap_insert': 0, 'heap_remove': 0, 'heap_retrieval': 0, 'heap_update': 0, "heap_total": 0,
                  'trace_length': 0, 'cost': 0, 'alignment_length': 0}
        '''
        # Choose one of the following align, then save the results in csv file for further analysis
        # Choice 1: the original algorithm in paper "Efficiently computing alignments algorithm
        # and datastructures" from Eindhoven University
        '''
        align1 = astar_tue_open_set_heapq.Inc_astar(event_log[case_index], model_net, model_im, model_fm)
        align = align1.apply(event_log[case_index], model_net, model_im, model_fm)

        # align1 = astar_tue_open_set_hotq.Inc_astar(event_log[case_index], model_net, model_im, model_fm)
        # align = align1.apply(event_log[case_index], model_net, model_im, model_fm)

        # align1 = astar_tue.Inc_astar(event_log[case_index], model_net, model_im, model_fm)
        # align = align1.apply(event_log[case_index], model_net, model_im, model_fm)

        # align1 = astar_bid.Inc_astar(event_log[case_index], model_net, model_im, model_fm)
        # align = align1.apply(event_log[case_index], model_net, model_im, model_fm)

        # align1 = astar_tue_pp.Inc_astar(event_log[case_index], model_net, model_im, model_fm)
        # align = align1.apply(event_log[case_index], model_net, model_im, model_fm)

        # print("\n",align)

        result['time_sum'] = align['time_sum']
        result['time_heuristic'] = align['time_heuristic']
        result['time_heap'] = align['time_heap']
        result['visited_states'] = align['visited_states']
        result['queued_states'] = align['queued_states']
        result['traversed_arcs'] = align['traversed_arcs']
        result['lp_solved'] = align['lp_solved']
        result['lp_for_ini_solved'] = align['lp_for_ini_solved']
        result['reopen_close'] = align['num_reopen_close']
        result['heap_insert'] = align['num_insert']
        result['heap_remove'] = align['num_removal']
        result['heap_retrieval'] = align['num_retrieval']
        result['heap_update'] = align['num_update']
        result['heap_total'] = align['heap_total']
        result['restart'] = align['restart']
        result['split_num'] = align['split_num']
        result['trace_length'] = align['trace_length']
        result['alignment_length'] = align['alignment_length']
        result['cost'] = align['cost']
        align_lst.append(align['alignment'])

        with open('F:\Thesis\data\data_sepsis\sepsis_tue_open_set_heapq_original.csv', 'a') as f_object:
            dictwriter_object = DictWriter(f_object, fieldnames=field_names)
            # Pass the dictionary as an argument to the Writerow()
            dictwriter_object.writerow(result)
            # Close the file object
            f_object.close()

    df = pd.read_csv('F:\Thesis\data\data_sepsis\sepsis_tue_open_set_heapq_original.csv')
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
                                                     "reopen_close",
                                                     "heap_insert",
                                                     "heap_remove",
                                                     "heap_update",
                                                     "heap_retrieval",
                                                     "heap_total",
                                                     "split_num",
                                                     "trace_length",
                                                     "alignment_length",
                                                     "cost"
                                                     ])
    df3 = pd.concat([df2, df]).reset_index(drop=True)
    df3['alignment'] = align_lst
    df3.to_csv('F:\Thesis\data\data_sepsis\sepsis_tue_open_set_heapq_original.csv', index=False)


if __name__ == "__main__":
    search()
