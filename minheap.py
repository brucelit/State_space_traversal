from copy import deepcopy


class MinHeap:
    def __init__(self):
        self.item_lst = []
        self.item_index = {}

    def peek(self):
        if len(self.item_lst) == 0:
            return False
        else:
            return self.item_lst[0]

    def swap(self, index1, index2):
        temp = deepcopy(self.item_lst[index1])
        self.item_lst[index1] = deepcopy(self.item_lst[index2])
        self.item_index[self.item_lst[index1].m] = index1
        self.item_lst[index2] = temp
        self.item_index[self.item_lst[index2].m] = index2

    # extract the first element from heap
    def poll(self):
        # get the least element and remove the index map
        min_item_lst = self.item_lst[0]
        del self.item_index[min_item_lst.m]
        # put the last element at index 0, update index map
        self.item_lst[0] = deepcopy(self.item_lst[len(self.item_lst) - 1])
        self.item_lst.remove(self.item_lst[len(self.item_lst) - 1])
        self.item_index[self.item_lst[0].m] = 0
        self.heapifyDown(0)
        return min_item_lst

    def remove_ele(self, marking):
        # get the index of marking to remove
        idx_to_remove = self.item_index[marking]
        del self.item_index[self.item_lst[idx_to_remove].m]
        self.item_lst.remove(self.item_lst[len(self.item_lst) - 1])

        # move the last marking to this index
        self.item_lst[idx_to_remove] = self.item_lst[len(self.item_lst) - 1]
        # remove the index of this marking
        self.item_index[self.item_lst[idx_to_remove].m] = idx_to_remove
        self.heapifyDown(idx_to_remove)

    def add(self, item):
        self.item_index[item.m] = len(self.item_lst)
        self.item_lst.append(item)
        self.heapifyUp()

    def hasParent(self, index):
        if index == 0:
            return False
        else:
            return True

    def getParentIndex(self, index):
        if index % 2 == 1:
            return int((index - 1) / 2)
        else:
            return int((index - 2) / 2)

    def heapifyUp(self):
        index = len(self.item_lst) - 1
        while (self.hasParent(index) and self.item_lst[self.getParentIndex(index)] > self.item_lst[index]):
            self.swap(self.getParentIndex(index), index)
            index = self.getParentIndex(index)

    def heapifyDown(self, idx):
        index = idx
        while (self.hasLeftChild(index)):
            smallerChildIndex = self.getLeftChildIndex(index)
            if self.hasRightChild(index) and self.rightChild(index) < self.leftChild(index):
                smallerChildIndex = self.getRightChildIndex(index)
            if self.item_lst[index] < self.item_lst[smallerChildIndex]:
                break
            else:
                self.swap(index, smallerChildIndex)
            index = smallerChildIndex

    def rightChild(self, index):
        return self.item_lst[index * 2 + 2]

    def leftChild(self, index):
        return self.item_lst[index * 2 + 1]

    def getLeftChildIndex(self, index):
        return index * 2 + 1

    def getRightChildIndex(self, index):
        return index * 2 + 2

    def hasLeftChild(self, index):
        if index * 2 + 1 >= len(self.item_lst):
            return False
        else:
            return True

    def hasRightChild(self, index):
        if index * 2 + 2 >= len(self.item_lst):
            return False
        else:
            return True

    def print_mh(self):
        for i in self.item_lst:
            print(i.m, i.f, i.g)
        print(self.item_index)

    def clear_heap(self):
        self.item_lst = []
        self.item_index = {}


class SearchTuple:
    def __init__(self, m, f, g):
        self.m = m
        self.f = f
        self.g = g

    def __lt__(self, other):
        if self.f != other.f:
            return self.f < other.f
        else:
            return self.g < other.g



if __name__ == '__main__':
    mh = MinHeap()
    mh.add(SearchTuple("a", 5, 6))
    mh.add(SearchTuple("b", 5, 8))
    mh.add(SearchTuple("c", 5, 5))
    mh.add(SearchTuple("d", 3, 6))
    mh.add(SearchTuple("e", 6, 6))
    mh.add(SearchTuple("f", 2, 6))
    mh.add(SearchTuple("g", 2, 4))
    mh.add(SearchTuple("h", 2, 6))
    mh.add(SearchTuple("i", 2, 7))
    mh.add(SearchTuple("j", 2, 5))
    mh.add(SearchTuple("k", 2, 4))
    mh.add(SearchTuple("l", 2, 2))
