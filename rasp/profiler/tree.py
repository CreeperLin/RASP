
def reset_timing_all(node):
    if not node['timer'] is None:
        node['timer'].reset()
    if not node['net_timer'] is None:
        node['net_timer'].reset()
    node['fwd'] = 0
    for k, c in node._children.items():
        reset_timing_all(c)

def _addindent(s_, numSpaces):
    s = s_.split('\n')
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s

class StatTreeNode():
    def __init__(self, name):
        self.name = name
        self._children = dict()
        self.stats = dict()
        self.parent = None
        self.get = self.stats.get
    
    def __setitem__(self, k, v):
        self.stats[k] = v

    def __getitem__(self, k):
        if not k in self.stats: return None
        return self.stats[k]

    def update_values(self, dct):
        for key, value in dct.items():
            self.stats[key] = value
    
    def find_child_index(self, child_name):
        assert isinstance(child_name, str)
        index = -1
        for i in range(len(self._children)):
            if child_name == self._children[i].name:
                index = i
        return index
    
    def add_child(self, name, child):
        assert isinstance(child, StatTreeNode)
        if not name in self._children:
            self._children[name] = child
            child.set_parent(self)
    
    def set_parent(self, p):
        self.parent = p
    
    @property
    def root(self):
        if self.parent is None: return self
        return self.parent.root

    @property
    def num_children(self):
        return len(self._children)

    def leaves(self, max_depth=-1):
        if len(self._children)==0 or max_depth == 0:
            yield self
        else:
            for k, c in self._children.items():
                for s in c.leaves(max_depth-1):
                    yield s
    
    def subnodes(self, max_depth=-1):
        if len(self._children)==0 or max_depth == 0:
            return
        for k, c in self._children.items():
            yield c
            for s in c.subnodes(max_depth-1):
                yield s
    
    def is_precendent(self, node):
        if self.parent is node:
            return True
        elif self.parent is None:
            return False
        return self.parent.is_precendent(node)
    
    def set_stat_all(self, name, val):
        self.stats[name] = val
        for k, c in self._children.items():
            c.set_stat_all(name, val)
    
    @property
    def children(self):
        for k, c in self._children.items():
            yield c

    def __repr__(self):
        self_repr = 'name: \'{}\'\n  stats: {}'.format(self.name, self.stats)
        child_repr = ''
        for cn, c in self._children.items():
            c_repr = _addindent(repr(c), 2)
            child_repr +=  '('+cn+'): ' +c_repr
        _repr = '(\n  '
        _repr += '%s\n  ' % self_repr
        _repr += 'nodes: '
        if len(child_repr)!=0: _repr += '[\n  %s]\n' % child_repr
        else: _repr += '[]\n'
        _repr += ')\n'
        return _repr