def _has_parent(idx):
    if idx == 0:
        return False
    else:
        return True


import timeit


class MinHeap:
    def __init__(self):
        self.lst = []
        self.idx = {}

    def swap(self, idx1, idx2):
        temp = self.lst[idx1]
        self.lst[idx1] = self.lst[idx2]
        self.idx[self.lst[idx1].m] = idx1
        self.lst[idx2] = temp
        self.idx[self.lst[idx2].m] = idx2
        # swap two elements for the index map and list position
        # self.lst[idx1], self.lst[idx2] = self.lst[idx2], self.lst[idx1]
        # self.idx[self.lst[idx1].m], self.idx[self.lst[idx2].m] = self.idx[self.lst[idx2].m], self.idx[self.lst[idx1].m]

    def heap_insert(self, marking):
        self.lst.append(marking)
        self.idx[marking.m] = len(self.lst)-1
        self._heap_heapify_up(len(self.lst)-1)

    def heap_pop(self):
        # print(len(self.lst), self.lst[0])
        if len(self.lst) > 1:
            marking_to_pop = self.lst[0]
            del self.idx[marking_to_pop.m]
            # update list and index
            self.lst[0] = self.lst[-1]
            self.idx[self.lst[0].m] = 0
            # remove the last element
            self.lst.pop()
            self._heap_heapify_down(0)
            return marking_to_pop
        else:
            marking_to_pop = self.lst[0]
            self.heap_clear()
            return marking_to_pop

    def heap_find(self, m):
        return True if m in self.idx else False

    def heap_get(self, m):
        return self.lst[self.idx[m]]

    # def heap_remove(self,m):
    #     idx_to_remove = self.idx[m]
    #     marking_to_pop = self.lst[self.idx[m]]
    #     if len(self.lst)-1 == 1:
    #         self.heap_clear()
    #     else:
    #         del self.idx[marking_to_pop.m]
    #         # update list and index
    #         self.idx[self.lst[-1].m] = idx_to_remove
    #         self.lst[idx_to_remove] = self.lst[-1]
    #         # remove the last element
    #         self.lst.pop()
    #         self._heap_heapify_down(idx_to_remove)

    def heap_update(self, marking):
        idx_to_update = self.idx[marking.m]
        self.lst[idx_to_update] = marking
        self._heap_heapify_down(idx_to_update)
        self._heap_heapify_up(idx_to_update)

    def heap_clear(self):
        self.lst.clear()
        self.idx.clear()

    def _heap_heapify_down(self, idx):
        while self._has_left_child(idx):
            smaller_index = idx*2+1
            if idx*2+2 < len(self.lst) and self.lst[idx*2+2] < self.lst[idx*2+1]:
                smaller_index = idx*2+2
            if self.lst[idx] < self.lst[smaller_index]:
                break
            else:
                self.swap(smaller_index, idx)
            idx = smaller_index

    def _heap_heapify_up(self, idx):
        index = idx
        while index != 0 and self.lst[index] < self.lst[int((index - 1) // 2)]:
            # self.swap(int((index - 1) // 2), index)
            parent_index = int((index - 1) // 2)
            # self.lst[index], self.lst[parent_index] = self.lst[parent_index], self.lst[index]
            # self.idx[self.lst[index].m], self.idx[self.lst[parent_index].m] = \
            #     self.idx[self.lst[parent_index].m], self.idx[self.lst[index].m]
            self.swap(parent_index, index)
            index = parent_index

    def _has_left_child(self, idx):
        if idx*2+1 < len(self.lst):
            return True
        else:
            return False

    def _get_parent(self, index):
        return int((index - 1) // 2)

    def _has_right_child(self, idx):
        if idx*2+2 < len(self.lst):
            return True
        else:
            return False

    def print_idx(self):
        print(self.idx)

    def print_lst(self):
        for i in self.lst:
            print(i.m)

    def get_len(self):
        return len(self.lst)

class Marking:
    def __init__(self, m, f, g):
        self.m = m
        self.f = f
        self.g = g

    def __lt__(self, other):
        if self.f < other.f:
            return True
        if self.g < other.g:
            return True


if __name__ == '__main__':
    m1 = Marking("a", 4, 4)
    m2 = Marking("b", 3, 3)
    m3 = Marking("c", 2, 2)
    m4 = Marking("d", 5, 5)
    m5 = Marking("e", 1, 1)
    m6 = Marking("f", 6, 6)

    heap1 = MinHeap()
    heap1.heap_insert(m1)
    heap1.heap_insert(m2)
    heap1.heap_insert(m3)
    heap1.heap_insert(m4)
    heap1.heap_insert(m5)
    heap1.heap_insert(m6)

    print("after insertion")
    heap1.print_idx()
    heap1.print_lst()

    heap1.heap_pop()
    print("remove the first")

    heap1.print_idx()
    heap1.print_lst()
    i = heap1.heap_get("a")
    i.f = 2
    i.g = 2
    heap1.heap_update(i)

    m7 = Marking("g", 7, 7)
    heap1.heap_insert(m7)
    heap1.print_idx()
    heap1.print_lst()
    print(heap1.heap_find("f"))