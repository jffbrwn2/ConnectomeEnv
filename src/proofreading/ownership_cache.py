"""Bounded caches for immutable atom ownership at one pinned timestamp."""
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
import numpy as np


class MissingOwnership(ValueError):
    """The source explicitly has no owner at this pinned historical time."""


class OwnershipCache:
    def __init__(self, identity, resolver, directory, memory_entries=65536, disk_entries=1000000, resolver_workers=1):
        if memory_entries<1 or disk_entries<1:raise ValueError('Positive cache capacities required')
        if resolver_workers<1:raise ValueError('Positive resolver worker count required')
        from concurrent.futures import ThreadPoolExecutor
        self.pool=ThreadPoolExecutor(max_workers=resolver_workers) if resolver_workers>1 else None
        self.identity=identity;self.resolver=resolver
        self.memory_entries=memory_entries;self.disk_entries=disk_entries
        self.memory=OrderedDict();self.lock=threading.RLock()
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        # SQLite writes must stay node-local on clusters with shared filesystems.
        # Entries are immutable, timestamp-pinned, and reloadable from resolver.
        local_root=os.environ.get('CONNECTOME_LOCAL_OWNERSHIP')
        if local_root:
            namespace=hashlib.sha256(str(Path(directory).resolve()).encode()).hexdigest()
            directory=Path(local_root)/namespace
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(directory/(key+'.sqlite'),timeout=30,check_same_thread=False)
        self.db.execute('CREATE TABLE IF NOT EXISTS owners (atom TEXT PRIMARY KEY, root TEXT NOT NULL, touched INTEGER NOT NULL)')
        self.db.execute('CREATE INDEX IF NOT EXISTS owners_age ON owners(touched)')
        self.requests=0;self.resolved_atoms=0;self.disk_hits=0

    def _remember(self, atom, root):
        self.memory[int(atom)]=int(root);self.memory.move_to_end(int(atom))
        while len(self.memory)>self.memory_entries:self.memory.popitem(last=False)

    def ensure(self, atoms, *, allow_missing=False):
        atoms=list(map(int,np.unique(np.asarray(atoms,dtype=np.uint64))))
        if any(a<=0 for a in atoms):raise ValueError('Foreground atoms required')
        with self.lock:
            result={a:self.memory[a] for a in atoms if a in self.memory}
            missing=[a for a in atoms if a not in result]
            for start in range(0,len(missing),800):
                batch=missing[start:start+800]
                marks=','.join('?' for _ in batch)
                for a,r in self.db.execute(f'SELECT atom,root FROM owners WHERE atom IN ({marks})',list(map(str,batch))):
                    result[int(a)]=int(r);self.disk_hits+=1
            missing=[a for a in missing if a not in result]
            pending={}
            batches=[np.asarray(missing[start:start+10000],dtype=np.uint64)
                     for start in range(0,len(missing),10000)]
            def resolve(batch):
                roots=np.asarray(self.resolver(batch))
                if roots.shape!=(len(batch),) or roots.dtype.kind not in 'iu' or np.any(roots<0):
                    raise ValueError('Missing or invalid pinned ownership; no guessed roots')
                if not allow_missing and np.any(roots==0):
                    raise MissingOwnership("Missing pinned ownership; no guessed roots")
                return dict(zip(map(int,batch),map(int,roots)))
            if self.pool is not None and len(batches)>1:
                futures=[self.pool.submit(resolve,batch) for batch in batches]
                from concurrent.futures import wait
                wait(futures)  # Settle siblings before rollback or retry after failure.
                results=[future.result() for future in futures]
            else:
                results=[resolve(batch) for batch in batches]
            for batch,result_batch in zip(batches,results):
                pending.update(result_batch);self.requests+=1;self.resolved_atoms+=len(batch)
            # A failed batch never commits a partial ownership result.
            result.update(pending)
            if not allow_missing and any(r==0 for r in result.values()):
                raise MissingOwnership("Missing pinned ownership; no guessed roots")
            now=time.time_ns()
            with self.db:
                self.db.executemany('INSERT OR REPLACE INTO owners VALUES (?,?,?)',
                                    [(str(a),str(r),now) for a,r in pending.items()])
                self.db.executemany('UPDATE owners SET touched=? WHERE atom=?',
                                    [(now,str(a)) for a in result if a not in pending and a not in self.memory])
                if pending:
                    count=self.db.execute('SELECT COUNT(*) FROM owners').fetchone()[0]
                    if count>self.disk_entries:
                        self.db.execute('DELETE FROM owners WHERE atom IN (SELECT atom FROM owners ORDER BY touched LIMIT ?)',(count-self.disk_entries,))
            for a,r in result.items():self._remember(a,r)
            return result

    def observe(self, atoms):
        return self.ensure(atoms,allow_missing=True)

    def owner(self, atom):
        atom=int(atom)
        with self.lock:
            if atom in self.memory:
                self.memory.move_to_end(atom)
                if self.memory[atom]==0:raise MissingOwnership("Missing pinned ownership; no guessed roots")
                return self.memory[atom]
        return self.ensure([atom])[atom]

    def stats(self):
        with self.lock:
            return {'resident_entries':len(self.memory),'memory_entry_budget':self.memory_entries,
                    'disk_entries':self.db.execute('SELECT COUNT(*) FROM owners').fetchone()[0],
                    'disk_entry_budget':self.disk_entries,'resolver_requests':self.requests,
                    'resolved_atoms':self.resolved_atoms,'disk_hits':self.disk_hits}

    def close(self):
        with self.lock:
            if self.pool is not None:self.pool.shutdown(wait=True)
            self.db.close()
