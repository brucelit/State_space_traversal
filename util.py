class State:
    def __init__(self, trust, f, g, h, marking, marking_tuple, pre_state, pre_transition, pre_trans_lst, last_sync, parikh_vector):
        self.trust = trust
        self.f = f
        self.g = g
        self.h = h
        self.pre_transition = pre_transition
        self.marking = marking
        self.marking_tuple = marking_tuple
        self.parikh_vector = parikh_vector
        self.pre_trans_lst = pre_trans_lst
        self.pre_state = pre_state
        self.last_sync = last_sync

    def __lt__(self, other):
        return (self.trust, self.f, self.g) < (other.trust, other.f, other.g)

    # def __lt__(self, other):
    #     if self.trust and not other.trust:
    #         return True
    #     elif self.f < other.f:
    #         return True
    #     elif other.f < self.f:
    #         return False
