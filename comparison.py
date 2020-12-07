from pm4py.algo.conformance.alignments.variants import state_equation_a_star
from pm4py.objects.log.importer.xes import importer as xes_importer
import time
from astar_implementation import construction

log = xes_importer.apply('E:\Thesis\sample_aeddeceh.xes')
model_net, model_im, model_fm = construction.construct_model_net()

print(log[0])
p1 = state_equation_a_star.Parameters
start_time = time.time()
dict1 = state_equation_a_star.apply(log[0], model_net, model_im, model_fm)
print("--- %s seconds ---" % (time.time() - start_time))
print(dict1)