from pm4py.objects.petri import synchronous_product
from astar_implementation import utilities, astar, initialization

if __name__ == '__main__':
    model_net, model_im, model_fm = utilities.construct_model_net()
    trace_net, trace_im, trace_fm = utilities.construct_trace_net_without_loop()
    sync_net, sync_im, sync_fm = synchronous_product.construct(trace_net, trace_im, trace_fm, model_net, model_im,
                                                               model_fm, '>>')
    aux_dict = initialization.initialize_aux_dict(sync_net, sync_im, sync_fm)
    split_lst = [None]
    align = astar.astar_with_split(sync_net, sync_im, sync_fm, aux_dict, split_lst)
