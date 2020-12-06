from pm4py.algo.conformance.alignments.variants import state_equation_a_star
from pm4py.objects.log.importer.xes import importer as xes_importer

from astar_implementation import utilities

log = xes_importer.apply('E:\Thesis\sample_addeg.xes')
model_net, model_im, model_fm = utilities.construct_model_net()

print(log[0])
p1 = state_equation_a_star.Parameters

dict1 = state_equation_a_star.apply(log[0], model_net, model_im, model_fm)
print(dict1)