import time
import func_timeout
import astar_pm4py
import astar_tue
import astar_bid
import astar_precompute
from csv import DictWriter
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri.importer.variants.pnml import import_net
from construction import precompute_forward, precompute_backward


def search():
    # Here to change the log file in dataset: the .xes file
    event_log = xes_importer.apply('C:\data\log_d.xes')
    # Here to change the model in dataset: the .pnml file
    model_net, model_im, model_fm = import_net('C:\data\model_d.pnml')
    # the colunm name in result csv file
    field_names = ['alignment',
                   'cost',
                   "visited_states",
                   "queued_states",
                   "traversed_arcs",
                   "lp_solved",
                   "restart",
                   "block_restart",
                   "trace_length",
                   "time"]

    # This incidence matrix is used to help precompute the split point
    incidence_matrix = astar_precompute.construct(model_net)

    # iterate every case in this xes log file
    for case_index, case in enumerate(event_log):
        start_time = time.time()

        try:
            '''
            # Choose one of the following align, then save the results in csv file for further analysis

            # Choice 1: the original algorithm in paper "Efficiently computing alignments algorithm
            # and datastructures" from Eindhoven University
            '''
            align = astar_tue.apply(case, model_net, model_im, model_fm)

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

            '''
            # Choice 4: the algorithm from pm4py based on paper "Computing Alignments of Event 
            # Data and Process Models“ given by RWTH University
            align = astar_pm4py.apply(case, model_net, model_im, model_fm)
            field_names = ['alignment',
                           'cost',
                           "visited_states",
                           "queued_states",
                           "traversed_arcs",
                           "lp_solved",
                           "time"]
            '''

            # save the running time for this conformance checking
            align['time'] = time.time() - start_time
            print(case_index, align['cost'])

            # save the conformance checking results as csv file
            with open('C:\data\c19_inc_astar_bid\c19_astar111.csv', 'a') as f_object:
                dictwriter_object = DictWriter(f_object, fieldnames=field_names)
                # Pass the dictionary as an argument to the Writerow()
                dictwriter_object.writerow(align)
                # Close the file object
                f_object.close()

        except func_timeout.exceptions.FunctionTimedOut:
            print("timeout", id)
            align = {'alignment': "??", 'cost': "??"}
            with open('C:\data\c19_inc_astar_bid\c19_astar111.csv', 'a') as f_object:
                dictwriter_object = DictWriter(f_object, fieldnames=field_names)
                # Pass the dictionary as an argument to the Writerow()
                dictwriter_object.writerow(align)
                # Close the file object
                f_object.close()


if __name__ == "__main__":
    search()
