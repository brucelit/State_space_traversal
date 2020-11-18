

class State:
    def __init__(self, f, g, h, t, m, m_tuple, p, solution_x,last_sync):
        self.f = f
        self.g = g
        self.h = h
        self.t = t
        self.m = m
        self.mt = m_tuple
        self.p = p
        self.solution_x = solution_x
        self.last_sync = last_sync

    def __lt__(self, other):
        if self.f < other.f:
            return True
        elif self.g < other.g:
            return True
        else:
            return self.h < other.h