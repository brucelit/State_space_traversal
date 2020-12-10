# from pm4py.objects.petri import synchronous_product
from astar_implementation import construction, astar, initialization, test, synchronous_product
import time
from memory_profiler import memory_usage


if __name__ == '__main__':
    model_net, model_im, model_fm = construction.construct_model_net()
    trace_net, trace_im, trace_fm = construction.construct_trace('abcebcegh')
    sync_net, sync_im, sync_fm, sync_index = synchronous_product.construct(trace_net, trace_im, trace_fm, model_net, model_im,
                                                               model_fm, '>>')
    aux_dict = initialization.initialize_aux_dict(sync_net, sync_im, sync_fm, sync_index)
    split_lst = {None: -1}
    start_time = time.time()
    align = test.astar_with_split(sync_net, sync_im, sync_fm, aux_dict, split_lst)
    print("--- %s seconds ---" % (time.time() - start_time))