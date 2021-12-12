import func_timeout
import statistics
from csv import DictWriter
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer.variants.pnml import import_net
from pm4py.algo.conformance.alignments.petri_net.variants import state_equation_a_star
import warnings
import pandas as pd

import astar_pm4py
import astar_tue
import astar_tue_latest
from tqdm import tqdm


def search():
    # Here to change the log file in dataset: the .xes file
    event_log = xes_importer.apply('F:\Thesis\data\BPIC19_2_101.xes')
    # Here to change the model in dataset: the .pnml file
    model_net, model_im, model_fm = import_net('F:\Thesis\data\BPIC19_2_IM.pnml')
    # the colunm name in result csv file
    field_names = ["time_sum",
                   "time_h",
                   "time_heapify",
                   "time_diff",
                   "lp_solved",
                   "visited_states",
                   "traversed_arcs",
                   "restart",
                   'cost']
    
    # df = pd.DataFrame(columns=field_names)
    # df.to_csv('F:\Thesis\data\BPIC19_2_astar_tue\BPIC19_2_astar_1210.csv', sep=',', index=False)

    # iterate every case in this xes log file
    for case_index in tqdm(range(len(event_log))):
        result2 = {}
        result = {'time_sum': [], 'time_h': [], 'time_heapify':[], 'time_diff': [], 'cost': [], 'visited_states': [],
                  'traversed_arcs': [], 'lp_solved': [], 'restart': []}
        try:
            # loop 5 times and get average
            for i in range(1):
                '''
                # Choose one of the following align, then save the results in csv file for further analysis
                # Choice 1: the original algorithm in paper "Efficiently computing alignments algorithm
                # and datastructures" from Eindhoven University
                '''
                # align = state_equation_a_star.apply(event_log[case_index], model_net, model_im, model_fm)
                # print(align)
                align1 = astar_tue.Inc_astar(event_log[case_index], model_net, model_im, model_fm)
                align = align1.apply(event_log[case_index], model_net, model_im, model_fm)
                print(align)
                # align = astar_tue_latest.apply(case, model_net, model_im, model_fm)
                # print(align)

                '''
                # Choice 2: the algorithm from in paper "Improving Alignment Computation
                # using Model-based Preprocessing" from Eindhoven University
                trace_lst1 = []
                trace_lst2 = []
                for event_index, event in enumerate(event_log[case_index]):
                    trace_lst1.append(event['concept:name'])
                    trace_lst2.insert(0, event['concept:name'])
                violate_lst_forward = precompute_forward(trace_lst1, ic)
                violate_lst_backward = precompute_backward(trace_lst2, ic)
                align = astar_precompute.apply(case, model_net, model_im, model_fm)
                '''

                '''
                # Choice 3: the algorithm from litian
                align = astar_bid.apply(case, model_net, model_im, model_fm)
                '''
                # align = astar_bid.apply(case, model_net, model_im, model_fm)

                # Choice 8: the algorithm from pm4py
                # align = state_equation_a_star.apply(event_log[case_index], model_net, model_im, model_fm)
                # print(align)
                # align = astar_pm4py.apply(case, model_net, model_im, model_fm)
                # print(align)

                result['time_sum'].append(align['time_sum'])
                result['time_h'].append(align['time_h'])
                result['time_diff'].append(align['time_diff'])
                result['time_heapify'].append(align['time_heapify'])
                result['cost'].append(align['cost'])
                result['visited_states'].append(align['visited_states'])
                result['traversed_arcs'].append(align['traversed_arcs'])
                result['lp_solved'].append(align['lp_solved'])
                result['restart'].append(align['restart'])

            result2['time_sum'] = statistics.mean(result['time_sum'])
            result2['time_h'] = statistics.mean(result['time_h'])
            result2['time_heapify'] = statistics.mean(result['time_heapify'])
            result2['time_diff'] = statistics.mean(result['time_diff'])
            result2['lp_solved'] = statistics.mean(result['lp_solved'])
            result2['visited_states'] = statistics.mean(result['visited_states'])
            result2['traversed_arcs'] = statistics.mean(result['traversed_arcs'])
            result2['cost'] = statistics.mean(result['cost'])
            result2['restart'] = statistics.mean(result['restart'])

            # with open('F:\Thesis\data\BPIC19_2_astar_tue\BPIC19_2_astar_1210.csv', 'a') as f_object:
            #     dictwriter_object = DictWriter(f_object, fieldnames=field_names)
            #     # Pass the dictionary as an argument to the Writerow()
            #     dictwriter_object.writerow(result2)
            #     # Close the file object
            #     f_object.close()

        except func_timeout.exceptions.FunctionTimedOut:
            print("timeout", id)
            align = {'alignment': "??", 'cost': "??"}
            with open('F:\Thesis\data\BPIC19_2_astar_tue\BPIC19_2_astar_1210.csv', 'a') as f_object:
                dictwriter_object = DictWriter(f_object, fieldnames=field_names)
                # Pass the dictionary as an argument to the Writerow()
                dictwriter_object.writerow(align)
                # Close the file object
                f_object.close()

    # df = pd.read_csv('F:\Thesis\data\BPIC19_2_astar_tue\BPIC19_2_astar_1210.csv')
    # total = df.sum()
    # df2 = pd.DataFrame([total.transpose()], columns=["time_sum",
    #                                                  "time_h",
    #                                                  "time_diff",
    #                                                  "lp_solved",
    #                                                  "visited_states",
    #                                                  "traversed_arcs",
    #                                                  "restart",
    #                                                  "cost"])
    # df3 = pd.concat([df2, df]).reset_index(drop=True)
    # df3.to_csv('F:\Thesis\data\BPIC19_2_astar_tue\BPIC19_2_astar_1210.csv', index=False)


if __name__ == "__main__":
    search()
