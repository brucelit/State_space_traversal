class State:
    def __init__(self, not_trust, f, g, h, marking, marking_tuple, pre_state, pre_transition, pre_trans_lst, last_sync,
                 parikh_vector):
        self.not_trust = not_trust
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
        return (self.f, self.not_trust, other.g) < (other.f, other.not_trust, self.g)
        # if self.f < other.f:
        #     return True
        # elif other.not_trust and not self.not_trust:
        #     return True
        # elif self.g > other.g:
        #     return True
        # elif self.f > other.f:
        #     return False
        # elif not other.not_trust and self.not_trust:
        #     return False
        # elif self.g < other.g:
        #     return False
        # elif self.g < other.g:
        #     return False
        # if self.f < other.f:
        #     return True
        # elif other.f < self.f:
        #     return False
        # elif not self.not_trust and other.not_trust:
        #     return True
        # else:
        #     return self.g > other.g

    def __gt__(self, other):
        # if self.f > other.f:
        #     return True
        # elif other.f > self.f:
        #     return False
        # elif self.not_trust and not other.not_trust:
        #     return True
        # else:
        #     return self.g < other.g
        return (self.f, self.not_trust, other.g) > (other.f, other.not_trust, self.g)

    def __eq__(self, other):
        return (self.f, self.not_trust, self.g) == (other.f, other.not_trust, other.g)
