from pm4py.objects.petri.importer.variants.pnml import import_net
from astar_implementation import construction, astar_v2, initialization, synchronous_product
from csv import DictWriter
import time
import func_timeout
import pickle
from pm4py.objects.petri.importer.variants.pnml import import_net
from pm4py.algo.conformance.alignments.variants import state_equation_a_star, state_equation_less_memory
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri.importer.variants.pnml import import_net

field_names = ['alignment', 'cost', 'queued_states', 'visited_states', 'traversed_arcs', 'recalculation', 'time', 'lp_solved']

# Import data set and petri net model
event_log = xes_importer.apply('F:\State_space_traversal\data\log_d.xes')
model_net, model_im, model_fm = import_net('F:\State_space_traversal\data\model_d.pnml')
for case_index, case in enumerate(event_log):
    try:
        start_time = time.time()
        align = state_equation_a_star.apply(event_log[case_index], model_net, model_im, model_fm)
        align['time'] = time.time() - start_time

        # save result to csv file
        with open('F:\State_space_traversal\cc_results\c20d_pm4py.csv', 'a') as f_object:
            dictwriter_object = DictWriter(f_object, fieldnames=field_names)
            # Pass the dictionary as an argument to the Writerow()
            dictwriter_object.writerow(align)
            # Close the file object
            f_object.close()

    # save results when function timeout
    except func_timeout.exceptions.FunctionTimedOut:
        print("timeout", id)
        align = {'alignment': "??", 'cost': "??", 'visited_states': "??", 'queued_states': "??", 'traversed_arcs': "??",
                'recalculation': "??",'time':'??'}
        with open('F:\State_space_traversal\cc_results\c20d_pm4py.csv', 'a') as f_object:
            dictwriter_object = DictWriter(f_object, fieldnames=field_names)
            # Pass the dictionary as an argument to the Writerow()
            dictwriter_object.writerow(align)
            # Close the file object
            f_object.close()