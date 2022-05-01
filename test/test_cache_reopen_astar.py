import timeit
import pandas as pd

from pathlib import Path
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as petri_importer
from tqdm import tqdm
from src.split_astar_cache_reopen import AlignmentWithCacheReopenAstar
from src.tools import SynchronousProduct


def compute_alignment(xes_file, pnml_file):
    """
    Compute alignments for event log with given model.
    Save alignments results and all metrics during computation in a csv file.

    Parameters
    ----------
    xes_file : .xes  file
               The xes file of the event log
    pnml_file : .pnml file
                The petri net model
    """

    event_log = xes_importer.apply(xes_file)
    model_net, model_im, model_fm = petri_importer.apply(pnml_file)
    # get log name and model name
    log_name = Path(log_path).stem
    model_name = Path(model_path).stem
    # define the column name in result file
    field_names = ['case_id',
                   'total',
                   'heuristic',
                   'queue',
                   'states',
                   'arcs',
                   'sum',
                   'num_insert',
                   'num_removal',
                   'num_update',
                   'simple_lp',
                   'complex_lp',
                   'restart',
                   'split_num',
                   'trace_length',
                   'alignment_length',
                   'cost',
                   'alignment']
    df = pd.DataFrame(columns=field_names)
    trace_variant_lst = {}
    # iterate every case in log
    for case_index in tqdm(range(len(event_log))):
        events_lst = []
        for event in event_log[case_index]:
            events_lst.append(event['concept:name'])
        trace_str = ''.join(events_lst)
        # if the sequence of events is met for the first time
        if trace_str not in trace_variant_lst:
            # construct synchronous product net
            sync_product = SynchronousProduct(event_log[case_index], model_net, model_im, model_fm)
            initial_marking, final_marking, cost_function, incidence_matrix, trace_sync, trace_log \
                = sync_product.construct_sync_product(event_log[case_index], model_net, model_im, model_fm)
            # compute alignment with split-pint-based algorithm + caching strategy + reopen method
            start_time = timeit.default_timer()
            ali_with_split_astar = AlignmentWithCacheReopenAstar(initial_marking, final_marking, cost_function,
                                                           incidence_matrix, trace_sync, trace_log)
            alignment_result = ali_with_split_astar.search()
            # get the total computation time
            alignment_result['total'] = timeit.default_timer() - start_time
            alignment_result['case_id'] = event_log[case_index].attributes['concept:name']
            trace_variant_lst[trace_str] = alignment_result
        else:
            alignment_result = trace_variant_lst[trace_str]
            alignment_result['case_id'] = event_log[case_index].attributes['concept:name']
        df = df.append(alignment_result, ignore_index=True)
    # The name of result csv file is of the form: 'log_name + model_name + algorithm type.csv'
    df.to_csv('../results/log=' + log_name + '&model=' + model_name + '&algorithm=cache_reopen_astar' + '.csv', index=False)


if __name__ == '__main__':
    log_path = '..\log\C20d_all.xes'
    model_path = '..\model\C20d_5.pnml'
    compute_alignment(log_path, model_path)
