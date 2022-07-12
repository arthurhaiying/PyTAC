import numpy as np
from tbn.tbn import TBN
from tbn.node import Node
import random
from sklearn.preprocessing import normalize

class GridNetwork:

    potentials = {}
    bn = TBN('grid')
    
    def __init__(self,m,n,s, mode='random'):
        self.m = m
        self.n = n
        self.s = s
        self.values = tuple('v%d' % i for i in range(self.s))
        for i in range(m):
            for j in range(n):
                self._init_potentials(i,j, mode)
        self._init_bn()
        

    def _init_potentials(self, i,j, mode):
        self.potentials[(i,j)] = []
        if 0 <= i-1:
            if mode == 'random':
                self.potentials[(i,j)].append(self._get_random_potential())
        if 0 <= j-1:
            if mode == 'random':
                self.potentials[(i,j)].append(self._get_random_potential())

    def _get_random_potential(self):
        r = np.random.rand(self.s,self.s)
        np.fill_diagonal(r, random.uniform(0, 1))
        return r

    def _init_bn(self):
        nodes = {}
        for i in range(self.m):
            for j in range(self.n):
                parents = []
                if 0 <= i-1:
                    parents.append(nodes[(i-1, j)])
                if 0 <= j-1:
                    parents.append(nodes[(i, j-1)])
                nd = Node('(%d,%d)' % (i,j), values=self.values, parents=parents, cpt=self._get_cpt(i,j))
                nodes[(i,j)] = nd
        for i in range(self.m):
            for j in range(self.n):
                self.bn.add(nodes[(i,j)])

    def _get_cpt(self, i, j):
        num_parents = len(self.potentials[(i,j)])
        if num_parents == 0:
            tmp = np.random.rand(self.s,)
            return (tmp/np.sum(tmp)).tolist()
        elif num_parents == 1:
            return normalize(self.potentials[(i,j)][0], axis=1, norm='l1').tolist()
        elif num_parents == 2:
            cpt = []
            for r in range(self.s):
                p1 = self.potentials[(i,j)][0]
                p2 = self.potentials[(i,j)][1]
                cpt.append(normalize(p1*p2[r], axis=1, norm='l1').tolist())
            return cpt

    def __repr__(self):
        st = 'Grid size: %dx%d \n' % (self.m,self.n)
        st += 'Potential size: %dx%d \n' % (self.s,self.s)
        for i in range(self.m):
            for j in range(self.n):
                st += 'Node(%d, %d): \n' % (i,j)
                st += 'Edge Potentials: \n'
                for p in self.potentials[(i,j)]:
                    st += str(p)
                    st += '\n'
                st += 'CPT: \n'
                st += str(self._get_cpt(i,j))
                st += '\n'
        return st

    def get_bn(self):
        return self.bn
        

if __name__ == '__main__':
    g = GridNetwork(2,3,2)
    print(g)

