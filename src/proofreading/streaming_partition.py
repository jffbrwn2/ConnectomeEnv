"""Persistent edit rules over a lazily discovered, pinned source partition.

Merges operate on segment identities, including atoms not downloaded yet.
Splits require complete parent membership. Undo copies share immutable rules,
so navigation and history do not copy a growing world-sized atom dictionary.
"""
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class Rule:
    parents: frozenset
    children: tuple
    assignments: object  # None for identity merge; immutable atom map for split.


class StreamingPartition:
    def __init__(self, source, rules=()):
        # Source contract: owner(atom), members(root), observed(root).
        # owner and members always refer to one pinned historical snapshot.
        self.source = source
        self.rules = rules

    def copy(self):
        return StreamingPartition(self.source,self.rules)

    def __getitem__(self, atom):
        atom=int(atom);sid=int(self.source.owner(atom))
        return self._apply(atom,sid)

    def display_owners(self, atoms):
        """Zero means explicitly unavailable for display, never an edit identity."""
        observer=getattr(self.source,'observe',None)
        if observer is None:return [self[int(a)] if a else 0 for a in atoms]
        foreground=[int(a) for a in atoms if a]
        lookup=observer(foreground)
        return [self._apply(int(a),lookup[int(a)]) if a and lookup[int(a)] else 0 for a in atoms]

    def _apply(self, atom, sid):
        if sid<=0:raise ValueError('Missing source ownership; cannot invent a segment')
        for rule in self.rules:
            if sid not in rule.parents:continue
            if rule.assignments is None:
                sid=rule.children[0]
            else:
                if atom not in rule.assignments:
                    raise ValueError('Source membership changed after a complete split')
                sid=rule.assignments[atom]
        return sid

    def roots_for(self, sid):
        sid=int(sid)
        lineage={}
        retired=set()
        for rule in self.rules:
            origins=set()
            for parent in rule.parents:origins.update(lineage.get(parent,{parent}))
            retired.update(rule.parents)
            for child in rule.children:lineage[child]=origins
        if sid in retired:return set()
        return lineage.get(sid,{sid})

    def split_roots(self):
        lineage={};affected=set()
        for rule in self.rules:
            roots=set().union(*(lineage.get(p,{p}) for p in rule.parents))
            if rule.assignments is not None:affected.update(roots)
            for child in rule.children:lineage[child]=roots
        return affected

    def atoms_for(self, sid, *, complete=False):
        sid=int(sid);result=set()
        for root in self.roots_for(sid):
            members=self.source.members(root) if complete else self.source.observed(root)
            result.update(int(a) for a in members if self._apply(int(a),root)==sid)
        return result

    def complete_atoms_for(self, sid):
        return self.atoms_for(sid,complete=True)

    def apply_groups(self, kind, parents, groups, children):
        parents=frozenset(map(int,parents));children=tuple(map(int,children))
        if not parents or len(set(children))!=len(children) or any(c<=0 for c in children):
            raise ValueError('Distinct positive child IDs and nonempty parents required')
        used={x for rule in self.rules for x in (*rule.parents,*rule.children)}
        if parents & set(children) or used & set(children):
            raise ValueError('Edit child IDs must be fresh')
        if kind=='merge':
            if len(parents)<2 or len(children)!=1:raise ValueError('Merge needs multiple parents and one child')
            if any(not self.atoms_for(p) for p in parents):raise ValueError('Merge requires live observed segments')
            assignments=None
        elif kind=='split':
            if len(parents)!=1 or len(groups)!=len(children) or len(children)<2:
                raise ValueError('Split needs one parent and multiple children')
            expected=self.atoms_for(next(iter(parents)),complete=True)
            assignments={}
            for group,child in zip(groups,children):
                if not group:raise ValueError('Split children must be nonempty')
                for atom in group:
                    atom=int(atom)
                    if atom in assignments:raise ValueError('Overlapping split children')
                    assignments[atom]=child
            if set(assignments)!=expected:
                raise ValueError('Split must cover complete parent membership, including unloaded atoms')
            assignments=MappingProxyType(assignments)
        else:raise ValueError('Unknown edit kind')
        self.rules += (Rule(parents,children,assignments),)

    def serialize(self):
        return [{'parents':sorted(map(str,r.parents)),'children':list(map(str,r.children)),
                 'assignments':None if r.assignments is None else
                 {str(a):str(s) for a,s in sorted(r.assignments.items())}} for r in self.rules]
