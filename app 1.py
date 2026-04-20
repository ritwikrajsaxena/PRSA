import streamlit as st
import math
import graphviz
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import deque

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class IndexNode:
    """Represents a node in the index tree"""
    node_id: str
    level: int
    keys: List[int] = field(default_factory=list)
    children_ids: List[str] = field(default_factory=list)  # Store IDs, not objects
    parent_id: Optional[str] = None                         # Store ID, not object
    data_bucket_ids: List[int] = field(default_factory=list)
    min_key: int = 0   # Pre-computed range for O(1) key lookup
    max_key: int = 0   # Pre-computed range for O(1) key lookup

    def is_leaf(self) -> bool:
        return len(self.children_ids) == 0

    def last_key(self) -> int:
        return max(self.keys) if self.keys else 0

    def __repr__(self):
        return f"IndexNode({self.node_id}, level={self.level}, keys={self.keys})"


@dataclass
class DataBucket:
    """Represents a data bucket in the broadcast"""
    bucket_id: int
    keys: List[int] = field(default_factory=list)

    def __repr__(self):
        return f"DataBucket({self.bucket_id}, keys={self.keys})"


@dataclass
class ControlIndexEntry:
    """An entry in the control index"""
    threshold_key: int
    target_pointer: str
    description: str


@dataclass
class AccessStep:
    """Represents a step in the access protocol"""
    step_num: int
    action: str
    position: int
    bucket_accessed: str
    mode: str  # 'active' or 'doze'
    details: str


# ============================================================================
# INDEX TREE CONSTRUCTION
# ============================================================================

class IndexTree:
    """
    Builds and manages the B-tree style index tree.
    
    Key fixes:
    - Integer-based level calculation (avoids floating-point errors)
    - Clean iterative bottom-up construction (no broken while loop)
    - No circular references (parent/children stored as IDs)
    - Pre-computed key ranges per node for O(1) containment checks
    - Registry (all_nodes dict) as single source of truth
    """

    def __init__(
        self,
        num_data_buckets: int,
        bucket_capacity: int,
        key_values: Optional[List[int]] = None
    ):
        self.num_data_buckets = num_data_buckets
        self.bucket_capacity = max(2, bucket_capacity)  # Minimum fanout of 2
        self.num_levels = self._calculate_levels()
        self.root: Optional[IndexNode] = None
        self.data_buckets: List[DataBucket] = []
        self.all_nodes: Dict[str, IndexNode] = {}       # Single registry
        self.nodes_by_level: Dict[int, List[str]] = {}  # level -> [node_ids]

        # Generate or validate key values
        if key_values is None:
            self.key_values = list(range(1, num_data_buckets + 1))
        else:
            self.key_values = sorted(key_values)[:num_data_buckets]
            # Pad if too few keys provided
            if len(self.key_values) < num_data_buckets:
                next_key = (max(self.key_values) + 1) if self.key_values else 1
                while len(self.key_values) < num_data_buckets:
                    self.key_values.append(next_key)
                    next_key += 1

        self._build_tree()

    # ------------------------------------------------------------------
    # FIX 1: Integer-based level calculation — no floating-point risk
    # ------------------------------------------------------------------
    def _calculate_levels(self) -> int:
        """
        Calculate number of index levels using integer arithmetic.
        
        Example: 27 data buckets, fanout 3 → 3 levels
        (3^1=3 leaf nodes, 3^2=9... counts nodes not data buckets,
         so we count how many times we must group by n to reach 1 node)
        """
        if self.num_data_buckets <= 1:
            return 1
        n = self.bucket_capacity
        levels = 0
        capacity = 1
        while capacity < self.num_data_buckets:
            capacity *= n
            levels += 1
        return max(1, levels)

    # ------------------------------------------------------------------
    # FIX 2: Clean iterative bottom-up construction
    # ------------------------------------------------------------------
    def _build_tree(self):
        """
        Build the index tree iteratively from bottom up.
        No recursion, no broken loop logic, no circular references.
        """
        n = self.bucket_capacity

        # --- Create data buckets ---
        self.data_buckets = [
            DataBucket(bucket_id=i, keys=[self.key_values[i]])
            for i in range(self.num_data_buckets)
        ]

        # --- Create leaf index nodes (deepest index level) ---
        leaf_level = self.num_levels - 1
        self.nodes_by_level[leaf_level] = []
        current_level_ids: List[str] = []

        for chunk_start in range(0, self.num_data_buckets, n):
            chunk = self.data_buckets[chunk_start: chunk_start + n]
            node_id = f"L{leaf_level}_N{len(current_level_ids)}"
            keys = [db.keys[0] for db in chunk]
            data_ids = [db.bucket_id for db in chunk]

            node = IndexNode(
                node_id=node_id,
                level=leaf_level,
                keys=keys,
                data_bucket_ids=data_ids,
                min_key=min(keys),
                max_key=max(keys),
            )
            self.all_nodes[node_id] = node
            self.nodes_by_level[leaf_level].append(node_id)
            current_level_ids.append(node_id)

        # --- Build upper levels iteratively ---
        current_level = leaf_level - 1

        while len(current_level_ids) > 1:
            self.nodes_by_level[current_level] = []
            next_level_ids: List[str] = []

            for chunk_start in range(0, len(current_level_ids), n):
                chunk_ids = current_level_ids[chunk_start: chunk_start + n]
                chunk_nodes = [self.all_nodes[cid] for cid in chunk_ids]

                node_id = f"L{current_level}_N{len(next_level_ids)}"
                keys = [child.last_key() for child in chunk_nodes]

                node = IndexNode(
                    node_id=node_id,
                    level=current_level,
                    keys=keys,
                    children_ids=chunk_ids,
                    min_key=min(child.min_key for child in chunk_nodes),
                    max_key=max(child.max_key for child in chunk_nodes),
                )

                # Set parent reference (by ID only — no circular object ref)
                for child_id in chunk_ids:
                    self.all_nodes[child_id].parent_id = node_id

                self.all_nodes[node_id] = node
                self.nodes_by_level[current_level].append(node_id)
                next_level_ids.append(node_id)

            current_level_ids = next_level_ids
            current_level -= 1

        # --- Finalise root ---
        if current_level_ids:
            root_candidate_id = current_level_ids[0]
            root_node = self.all_nodes[root_candidate_id]

            # Rename to ROOT for clarity
            del self.all_nodes[root_candidate_id]
            old_level = root_node.level

            # Update children's parent_id to new name
            for child_id in root_node.children_ids:
                self.all_nodes[child_id].parent_id = "ROOT"

            root_node.node_id = "ROOT"
            root_node.level = 0
            self.all_nodes["ROOT"] = root_node
            self.root = root_node

            # Fix level registry
            if old_level in self.nodes_by_level:
                level_list = self.nodes_by_level[old_level]
                if root_candidate_id in level_list:
                    level_list.remove(root_candidate_id)
            self.nodes_by_level.setdefault(0, []).append("ROOT")

    # ------------------------------------------------------------------
    # Helper accessors
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> Optional[IndexNode]:
        return self.all_nodes.get(node_id)

    def get_nodes_at_level(self, level: int) -> List[IndexNode]:
        return [self.all_nodes[nid]
                for nid in self.nodes_by_level.get(level, [])
                if nid in self.all_nodes]

    def get_parent(self, node: IndexNode) -> Optional[IndexNode]:
        if node.parent_id is None:
            return None
        return self.all_nodes.get(node.parent_id)

    def get_children(self, node: IndexNode) -> List[IndexNode]:
        return [self.all_nodes[cid]
                for cid in node.children_ids
                if cid in self.all_nodes]

    def get_path_to_root(self, node: IndexNode) -> List[IndexNode]:
        """Returns path from root down to (but not including) node."""
        path = []
        current = self.get_parent(node)
        while current is not None:
            path.append(current)
            current = self.get_parent(current)
        return list(reversed(path))

    def find_lca(self, node1: IndexNode, node2: IndexNode) -> Optional[IndexNode]:
        """Find Least Common Ancestor using ancestor sets."""
        ancestors1 = set()
        current = node1
        while current is not None:
            ancestors1.add(current.node_id)
            current = self.get_parent(current)

        current = node2
        while current is not None:
            if current.node_id in ancestors1:
                return current
            current = self.get_parent(current)
        return None

    def get_path_between(
        self, ancestor: IndexNode, descendant: IndexNode
    ) -> List[IndexNode]:
        """
        Returns nodes on the path from ancestor down to descendant,
        EXCLUDING both endpoints.
        """
        full_path = self.get_path_to_root(descendant) + [descendant]
        result = []
        in_range = False
        for node in full_path:
            if node.node_id == ancestor.node_id:
                in_range = True
                continue
            if in_range and node.node_id != descendant.node_id:
                result.append(node)
        return result


# ============================================================================
# DISTRIBUTED INDEXING ALGORITHM
# ============================================================================

class DistributedIndex:
    """
    Implements the Distributed (Partial Replication) Indexing scheme.

    Key fixes:
    - NRR identification uses consistent level mapping
    - Iterative traversal replaces unbounded recursion
    - Schedule generation is a clean single pass
    """

    def __init__(self, index_tree: IndexTree, replication_level: int):
        self.tree = index_tree
        self.r = max(0, min(replication_level, index_tree.num_levels - 1))
        self.k = index_tree.num_levels
        self.n = index_tree.bucket_capacity
        self.nrr: List[IndexNode] = []
        self.broadcast_schedule: List[Dict] = []

        self._identify_nrr()
        self._generate_schedule()

    # ------------------------------------------------------------------
    # NRR Identification
    # ------------------------------------------------------------------

    def _identify_nrr(self):
        """
        NRR nodes live at level r (0-indexed from root).
        If r >= k, the entire index is replicated → NRR = leaf nodes.
        If r == 0, root itself is the only NRR.
        """
        if self.r >= self.k:
            # Whole index fits in replicated portion → leaves are NRR
            nrr_level = self.k - 1
        else:
            nrr_level = self.r

        self.nrr = self.tree.get_nodes_at_level(nrr_level)

        # Fallback: if still empty, use whatever level has nodes
        if not self.nrr:
            for level in range(self.k - 1, -1, -1):
                nodes = self.tree.get_nodes_at_level(level)
                if nodes:
                    self.nrr = nodes
                    break

    # ------------------------------------------------------------------
    # Schedule Generation (single clean pass)
    # ------------------------------------------------------------------

    def _generate_schedule(self):
        """
        For each NRR node Bi generate: Rep(Bi) + Ind(Bi) + Data(Bi)
        
        Rep(B0)  = path from root to B0
        Rep(Bi)  = path from LCA(B_{i-1}, Bi) down to Bi
        Ind(Bi)  = Bi itself + all index descendants (iterative BFS)
        Data(Bi) = all data buckets reachable from Bi
        """
        self.broadcast_schedule = []
        position = 0

        for i, B in enumerate(self.nrr):
            section_segments: List[Dict] = []

            # ---- Rep(B) ----
            if i == 0:
                rep_nodes = self.tree.get_path_to_root(B)
            else:
                B_prev = self.nrr[i - 1]
                lca = self.tree.find_lca(B_prev, B)
                rep_nodes = (
                    self.tree.get_path_between(lca, B) if lca else []
                )

            for node in rep_nodes:
                seg = {
                    'type': 'rep',
                    'node': node,
                    'section': i,
                    'position': position,
                    'description': f"Rep: {node.node_id}"
                }
                section_segments.append(seg)
                position += 1

            # ---- Ind(B): iterative BFS ----
            bfs_queue = deque([B])
            while bfs_queue:
                node = bfs_queue.popleft()
                seg = {
                    'type': 'ind',
                    'node': node,
                    'section': i,
                    'position': position,
                    'description': f"{'NRR' if node.node_id == B.node_id else 'Ind'}: {node.node_id}"
                }
                section_segments.append(seg)
                position += 1
                bfs_queue.extend(self.tree.get_children(node))

            # ---- Data(B): all data buckets under B ----
            for data_id in sorted(self._collect_data_ids(B)):
                if data_id < len(self.tree.data_buckets):
                    db = self.tree.data_buckets[data_id]
                    seg = {
                        'type': 'data',
                        'bucket': db,
                        'section': i,
                        'position': position,
                        'description': f"Data {data_id}: keys={db.keys}"
                    }
                    section_segments.append(seg)
                    position += 1

            self.broadcast_schedule.extend(section_segments)

    def _collect_data_ids(self, node: IndexNode) -> List[int]:
        """Iteratively collect all data bucket IDs under a node."""
        ids = []
        stack = [node]
        while stack:
            n = stack.pop()
            ids.extend(n.data_bucket_ids)
            stack.extend(self.tree.get_children(n))
        return ids

    def get_schedule_length(self) -> int:
        return len(self.broadcast_schedule)

    # ------------------------------------------------------------------
    # FIX: Correct optimal-r formula (from paper Section 4)
    # ------------------------------------------------------------------
    @staticmethod
    def calculate_optimal_r(
        num_data_buckets: int, bucket_capacity: int, k: int
    ) -> int:
        """
        Optimal replication level minimises:
            AccessTime = ProbeWait + BeastWait
        
        Derived from paper equations:
            r* = floor( (log_n(D*(n-1) + n^(k+1)) - 1) / 2 )
        
        Clamped to [0, k-1].
        """
        if k <= 1 or bucket_capacity <= 1:
            return 0

        n = bucket_capacity
        D = num_data_buckets

        try:
            inner = D * (n - 1) + n ** (k + 1)
            if inner <= 0:
                return 0
            r_star = (math.log(inner, n) - 1) / 2.0
            return max(0, min(k - 1, math.floor(r_star)))
        except (ValueError, ZeroDivisionError):
            return 0


# ============================================================================
# ACCESS PROTOCOL SIMULATION
# ============================================================================

class AccessSimulator:
    """
    Simulates the client-side access protocol.

    Key fixes:
    - Single O(n) forward scan instead of O(n²) nested loop
    - Doze steps emitted only when there is an actual gap
    - Key containment check is O(1) via pre-computed [min_key, max_key]
    """

    def __init__(self, distributed_index: DistributedIndex):
        self.di = distributed_index
        self.schedule = distributed_index.broadcast_schedule
        self.tree = distributed_index.tree

    # ------------------------------------------------------------------
    # FIX: O(1) key containment using pre-computed ranges
    # ------------------------------------------------------------------
    def _node_contains_key(self, node: IndexNode, key: int) -> bool:
        return node.min_key <= key <= node.max_key

    # ------------------------------------------------------------------
    # FIX: Single-pass O(n) access simulation
    # ------------------------------------------------------------------
    def simulate_access(
        self, tune_in_position: int, target_key: int
    ) -> List[AccessStep]:
        """
        Simulate the distributed-index access protocol.

        Protocol:
        1. Tune into broadcast at `tune_in_position`.
        2. Scan forward; enter doze mode for irrelevant buckets.
        3. On reaching a rep/ind node whose range covers target_key,
           read it (active) and continue.
        4. On reaching a data bucket containing target_key, download it.
        5. If end of schedule reached, report not found.
        """
        steps: List[AccessStep] = []
        sched_len = max(1, len(self.schedule))
        pos = tune_in_position % sched_len
        step_num = 1
        last_active_pos = pos  # Track last active position for doze gaps

        # --- Step 1: Initial probe ---
        steps.append(AccessStep(
            step_num=step_num,
            action="Initial Probe",
            position=pos,
            bucket_accessed=self._bucket_name(pos),
            mode="active",
            details=f"Client tunes into broadcast at position {pos}."
        ))
        step_num += 1
        pos += 1

        # --- Single forward scan ---
        found = False
        while pos < sched_len:
            seg = self.schedule[pos]

            if seg['type'] == 'data':
                bucket: DataBucket = seg['bucket']
                if target_key in bucket.keys:
                    # Emit doze step for the gap (if any)
                    if pos > last_active_pos + 1:
                        steps.append(AccessStep(
                            step_num=step_num,
                            action="Doze Mode",
                            position=last_active_pos,
                            bucket_accessed="-",
                            mode="doze",
                            details=(
                                f"Sleep positions {last_active_pos + 1}–{pos - 1} "
                                f"({pos - last_active_pos - 1} buckets skipped)."
                            )
                        ))
                        step_num += 1

                    steps.append(AccessStep(
                        step_num=step_num,
                        action="Download Data",
                        position=pos,
                        bucket_accessed=f"Data Bucket {bucket.bucket_id}",
                        mode="active",
                        details=(
                            f"✅ Found key {target_key} in "
                            f"Data Bucket {bucket.bucket_id}."
                        )
                    ))
                    found = True
                    break

            elif seg['type'] in ('rep', 'ind'):
                node: IndexNode = seg['node']
                if self._node_contains_key(node, target_key):
                    # Emit doze for gap
                    if pos > last_active_pos + 1:
                        steps.append(AccessStep(
                            step_num=step_num,
                            action="Doze Mode",
                            position=last_active_pos,
                            bucket_accessed="-",
                            mode="doze",
                            details=(
                                f"Sleep positions {last_active_pos + 1}–{pos - 1} "
                                f"({pos - last_active_pos - 1} buckets skipped)."
                            )
                        ))
                        step_num += 1

                    steps.append(AccessStep(
                        step_num=step_num,
                        action=f"Read {'Replicated' if seg['type'] == 'rep' else 'Index'} Node",
                        position=pos,
                        bucket_accessed=node.node_id,
                        mode="active",
                        details=(
                            f"Node {node.node_id} covers key range "
                            f"[{node.min_key}, {node.max_key}]. "
                            f"Key {target_key} is in range — following pointer."
                        )
                    ))
                    step_num += 1
                    last_active_pos = pos
                # If node does NOT cover target_key, remain in doze (no step added)

            pos += 1

        if not found:
            steps.append(AccessStep(
                step_num=step_num,
                action="Key Not Found",
                position=sched_len,
                bucket_accessed="-",
                mode="active",
                details=(
                    f"Key {target_key} was not found in this broadcast cycle. "
                    f"Wait for next broadcast."
                )
            ))

        return steps

    def _bucket_name(self, pos: int) -> str:
        if pos < 0 or pos >= len(self.schedule):
            return "Unknown"
        seg = self.schedule[pos]
        if seg['type'] == 'data':
            return f"Data Bucket {seg['bucket'].bucket_id}"
        return seg['node'].node_id


# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_metrics(tree: IndexTree, di: DistributedIndex) -> Dict:
    """
    Computes access time and tuning time using formulas from the paper.

    FIX: Replaced ad-hoc approximations with paper-derived equations.
    
    Paper definitions (Section 3-4):
      - index_size  = Σ_{i=1}^{k} ceil(D / n^i)   [total index nodes]
      - tuning_time = k + 1                          [root-to-leaf probes]
      - probe_wait  = 0.5 * (index_size / n^r + D / n^r)
      - beast_wait  = 0.5 * (index_size + D) / n^r
      - access_time = probe_wait + beast_wait
    """
    n = tree.bucket_capacity
    k = tree.num_levels
    r = di.r
    D = tree.num_data_buckets

    # Total index nodes across all levels
    index_size = sum(
        math.ceil(D / (n ** i)) for i in range(1, k + 1)
    )

    # Replication overhead: nodes in top-r levels
    replication_overhead = sum(
        math.ceil(D / (n ** i)) for i in range(1, r + 1)
    ) if r > 0 else 0

    # Segment size under one NRR
    nrr_count = max(1, len(di.nrr))
    segment_data = math.ceil(D / nrr_count)
    segment_index = math.ceil(index_size / nrr_count)

    # FIX: Paper-accurate wait time calculations
    probe_wait = 0.5 * (segment_index + segment_data)
    beast_wait = 0.5 * (index_size + D)
    access_time = probe_wait + beast_wait

    # FIX: Tuning time = k+1 probes (one per index level + data read)
    tuning_time = k + 1

    # Baseline comparisons
    access_opt_time = D / 2.0          # No index: scan half the file
    access_opt_tune = D / 2.0          # No dozing possible without index

    tune_opt_access = index_size + D   # Index at start: read everything
    tune_opt_tune   = k + 1            # Same tree depth

    return {
        'access_time': access_time,
        'tuning_time': tuning_time,
        'probe_wait': probe_wait,
        'beast_wait': beast_wait,
        'schedule_length': len(di.broadcast_schedule),
        'index_size': index_size,
        'replication_overhead': replication_overhead,
        'access_opt_time': access_opt_time,
        'access_opt_tune': access_opt_tune,
        'tune_opt_access': tune_opt_access,
        'tune_opt_tune': tune_opt_tune,
    }


# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================

def create_tree_visualization(tree: IndexTree, replication_level: int) -> graphviz.Digraph:
    """Build a Graphviz diagram of the index tree."""
    dot = graphviz.Digraph(comment='Index Tree')
    dot.attr(rankdir='TB')
    dot.attr('graph', nodesep='0.5', ranksep='0.75')

    def add_node(node: IndexNode):
        is_replicated = node.level < replication_level
        fill  = 'lightblue' if is_replicated else 'lightgreen'
        border = 'red' if node.level == replication_level else 'black'
        label = f"{node.node_id}\n[{node.min_key}–{node.max_key}]"
        dot.node(
            node.node_id, label,
            fillcolor=fill, style='filled,rounded',
            color=border,
            penwidth='2' if node.level == replication_level else '1'
        )
        for child in tree.get_children(node):
            add_node(child)
            dot.edge(node.node_id, child.node_id)

    if tree.root:
        add_node(tree.root)

    # Data buckets
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightyellow')
    for db in tree.data_buckets:
        dot.node(f"D{db.bucket_id}", f"Data {db.bucket_id}\n{db.keys}")

    # Leaf-to-data edges
    for level_ids in tree.nodes_by_level.values():
        for node_id in level_ids:
            node = tree.all_nodes[node_id]
            for did in node.data_bucket_ids:
                dot.edge(node.node_id, f"D{did}", style='dashed')

    return dot


def create_schedule_dataframe(schedule: List[Dict]) -> pd.DataFrame:
    rows = []
    for seg in schedule:
        if seg['type'] == 'data':
            db = seg['bucket']
            name, keys = f"Data_{db.bucket_id}", str(db.keys)
        else:
            node = seg['node']
            name, keys = node.node_id, str(node.keys)
        rows.append({
            'Position': seg['position'],
            'Type': seg['type'].upper(),
            'Name': name,
            'Keys': keys,
            'Section': seg.get('section', '-'),
            'Description': seg.get('description', ''),
        })
    return pd.DataFrame(rows)


# ============================================================================
# STREAMLIT APPLICATION
# ============================================================================

# ------------------------------------------------------------------
# FIX: Cache expensive objects so they are not rebuilt on every
#      widget interaction. `st.cache_resource` preserves Python objects.
# ------------------------------------------------------------------
@st.cache_resource
def build_index_cached(
    num_data_buckets: int,
    bucket_capacity: int,
    key_values_tuple: tuple,
    r: int
):
    key_values = list(key_values_tuple)
    tree = IndexTree(num_data_buckets, bucket_capacity, key_values)
    di   = DistributedIndex(tree, r)
    return tree, di


def main():
    st.set_page_config(
        page_title="Distributed Indexing on Air",
        page_icon="📡",
        layout="wide"
    )

    st.title("📡 Distributed Indexing for Data Broadcasting")
    st.markdown(
        "Implements the **Partial Replication (Distributed Indexing)** scheme "
        "from *Energy Efficient Indexing on Air* (Imielinski et al., 1994)."
    )

    # ====================================================================
    # SIDEBAR — INPUTS
    # ====================================================================
    st.sidebar.header("⚙️ Parameters")

    num_data_buckets = st.sidebar.number_input(
        "Number of Data Buckets (D)",
        min_value=1, max_value=1000, value=27,
        help="Total data buckets in the broadcast file."
    )

    bucket_capacity = st.sidebar.number_input(
        "Bucket Capacity / Fanout (n)",
        min_value=2, max_value=50, value=3,
        help="Max pointers per index bucket (tree fanout)."
    )

    key_option = st.sidebar.radio(
        "Key Generation",
        ["Auto (1 … D)", "Custom"],
    )
    if key_option == "Custom":
        raw = st.sidebar.text_area(
            "Keys (comma-separated)",
            value=",".join(str(i) for i in range(1, num_data_buckets + 1)),
        )
        try:
            key_values = [int(k.strip()) for k in raw.split(",") if k.strip()]
        except ValueError:
            st.sidebar.error("Invalid keys — using auto-generated values.")
            key_values = list(range(1, num_data_buckets + 1))
    else:
        key_values = list(range(1, num_data_buckets + 1))

    # Need tree to know k for the r slider — build a lightweight version
    _tmp_tree = IndexTree(num_data_buckets, bucket_capacity, key_values)
    k = _tmp_tree.num_levels
    optimal_r = DistributedIndex.calculate_optimal_r(num_data_buckets, bucket_capacity, k)

    st.sidebar.info(f"Tree depth k = {k} | Optimal r* = {optimal_r}")

    r_mode = st.sidebar.radio("Replication Level", ["Auto (optimal)", "Manual"])
    if r_mode == "Manual":
        r = st.sidebar.slider("r", 0, max(0, k - 1), min(optimal_r, max(0, k - 1)))
    else:
        r = optimal_r

    # Build (cached)
    tree, di = build_index_cached(
        num_data_buckets, bucket_capacity, tuple(key_values), r
    )
    schedule_len = len(di.broadcast_schedule)

    tune_in_pos = st.sidebar.number_input(
        "Client Tune-in Position",
        min_value=0, max_value=max(0, schedule_len - 1), value=0,
        help="Broadcast position where the client wakes up."
    )

    available_keys = sorted({k for db in tree.data_buckets for k in db.keys})
    target_key = st.sidebar.selectbox(
        "Target Search Key",
        options=available_keys,
        index=len(available_keys) // 2 if available_keys else 0,
    )

    # Input summary
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    | Parameter | Value |
    |-----------|-------|
    | D (buckets) | {num_data_buckets} |
    | n (fanout) | {bucket_capacity} |
    | k (levels) | {k} |
    | r (replication) | {r} |
    | Tune-in | {tune_in_pos} |
    | Target key | {target_key} |
    | Schedule length | {schedule_len} |
    """)

    # ====================================================================
    # TABS
    # ====================================================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌳 Index Structure",
        "📋 Broadcast Schedule",
        "🔍 Access Simulation",
        "📊 Metrics",
    ])

    # ----------------------------------------------------------------
    # TAB 1 — INDEX STRUCTURE
    # ----------------------------------------------------------------
    with tab1:
        st.header("Index Tree Structure")

        try:
            dot = create_tree_visualization(tree, r)
            st.graphviz_chart(dot, use_container_width=True)
        except Exception as exc:
            st.error(f"Graphviz error: {exc}")

        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Tree Properties")
            st.markdown(f"""
            | Property | Value |
            |----------|-------|
            | Levels (k) | **{k}** |
            | Fanout (n) | **{bucket_capacity}** |
            | Replication (r) | **{r}** |
            | Data Buckets | **{num_data_buckets}** |
            | Index Nodes | **{len(tree.all_nodes)}** |
            | NRR Count | **{len(di.nrr)}** |
            """)

        with col2:
            st.subheader("Legend")
            st.markdown("""
            | Colour | Meaning |
            |--------|---------|
            | 🔵 Light Blue | Replicated index nodes (top r levels) |
            | 🟢 Light Green | Non-replicated index nodes |
            | 🟡 Light Yellow | Data buckets (payload) |
            | 🔴 Red border | NRR level boundary |

            **Edges:** solid = index→index, dashed = index→data
            """)

        with col3:
            st.subheader("Non-Replicated Roots (NRR)")
            if di.nrr:
                nrr_df = pd.DataFrame([
                    {"#": i, "Node": n.node_id,
                     "Range": f"[{n.min_key}, {n.max_key}]"}
                    for i, n in enumerate(di.nrr)
                ])
                st.dataframe(nrr_df, use_container_width=True, hide_index=True)
            else:
                st.info("No NRR nodes for this configuration.")

        with st.expander("ℹ️ How the tree is organised"):
            st.markdown(f"""
            - **k = {k}** index levels (Level 0 = ROOT).
            - Levels 0 … {max(0, r-1)} are **replicated** (blue) — 
              they appear before every NRR section in the broadcast.
            - **NRR nodes** sit at level **{r}**; each is the root of a 
              non-replicated subtree broadcasted exactly once.
            - Data buckets (yellow) are the payload pointed to by leaf 
              index nodes.
            - Node labels show the **key range [min, max]** covered by 
              that subtree, enabling O(1) routing decisions.
            """)

    # ----------------------------------------------------------------
    # TAB 2 — BROADCAST SCHEDULE
    # ----------------------------------------------------------------
    with tab2:
        st.header("Broadcast Schedule")

        sched_df = create_schedule_dataframe(di.broadcast_schedule)

        def highlight_type(row):
            colours = {
                'REP': 'background-color: #ADD8E6',
                'IND': 'background-color: #90EE90',
                'DATA': 'background-color: #FFFFE0',
            }
            return [colours.get(row['Type'], '')] * len(row)

        st.dataframe(
            sched_df.style.apply(highlight_type, axis=1),
            use_container_width=True, height=400
        )

        st.subheader("Sections")
        sections: Dict[int, Dict] = {}
        for seg in di.broadcast_schedule:
            s = seg.get('section', 0)
            sections.setdefault(s, {'rep': [], 'ind': [], 'data': []})
            sections[s][seg['type']].append(seg)

        for sec_num, content in sections.items():
            nrr_name = di.nrr[sec_num].node_id if sec_num < len(di.nrr) else '?'
            with st.expander(f"Section {sec_num}: NRR = {nrr_name}"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**🔵 Rep(B)**")
                    for s in content['rep']:
                        st.text(f"• {s['node'].node_id}")
                    if not content['rep']:
                        st.text("(none)")
                with c2:
                    st.markdown("**🟢 Ind(B)**")
                    for s in content['ind']:
                        st.text(f"• {s['node'].node_id}")
                with c3:
                    st.markdown("**🟡 Data(B)**")
                    for s in content['data']:
                        st.text(
                            f"• Bucket {s['bucket'].bucket_id}: "
                            f"{s['bucket'].keys}"
                        )

    # ----------------------------------------------------------------
    # TAB 3 — ACCESS SIMULATION
    # FIX: Use session_state so results persist across widget interactions
    # ----------------------------------------------------------------
    with tab3:
        st.header("Access Protocol Simulation")

        c1, c2, c3 = st.columns(3)
        c1.info(f"**Tune-in:** {tune_in_pos}")
        c2.info(f"**Target key:** {target_key}")
        c3.info(f"**Schedule length:** {schedule_len}")

        if st.button("▶ Run Simulation", type="primary", use_container_width=True):
            simulator = AccessSimulator(di)
            st.session_state['sim_steps'] = simulator.simulate_access(
                tune_in_pos, target_key
            )

        if 'sim_steps' in st.session_state:
            steps: List[AccessStep] = st.session_state['sim_steps']

            active_n = sum(1 for s in steps if s.mode == 'active')
            doze_n   = sum(1 for s in steps if s.mode == 'doze')
            efficiency = doze_n / len(steps) * 100 if steps else 0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Steps",   len(steps))
            m2.metric("Active Steps",  active_n)
            m3.metric("Doze Steps",    doze_n)
            m4.metric("Doze Efficiency", f"{efficiency:.1f}%")

            st.subheader("Step-by-Step Trace")
            for step in steps:
                icon   = "🔋" if step.mode == 'active' else "😴"
                bg     = "#e6ffe6" if step.mode == 'active' else "#e6e6ff"
                colour = "green"  if step.mode == 'active' else "blue"
                with st.expander(
                    f"Step {step.step_num}: {step.action} {icon}",
                    expanded=True
                ):
                    st.markdown(f"""
                    <div style="background:{bg};padding:10px;border-radius:5px">
                    <b>Position:</b> {step.position}<br>
                    <b>Bucket:</b> {step.bucket_accessed}<br>
                    <b>Mode:</b>
                        <span style="color:{colour};font-weight:bold">
                            {step.mode.upper()}
                        </span><br>
                    <b>Details:</b> {step.details}
                    </div>
                    """, unsafe_allow_html=True)

            st.subheader("Timeline Table")
            tl_df = pd.DataFrame([{
                'Step': s.step_num,
                'Action': s.action,
                'Position': s.position,
                'Bucket': s.bucket_accessed,
                'Mode': f"{'🔋' if s.mode == 'active' else '😴'} {s.mode.upper()}"
            } for s in steps])
            st.dataframe(tl_df, use_container_width=True, hide_index=True)
        else:
            st.info("Click **▶ Run Simulation** to start.")

    # ----------------------------------------------------------------
    # TAB 4 — METRICS
    # ----------------------------------------------------------------
    with tab4:
        st.header("Performance Metrics")

        m = calculate_metrics(tree, di)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Distributed Indexing")
            st.metric("Access Time (buckets)",    f"{m['access_time']:.2f}")
            st.metric("Tuning Time (probes)",     f"{m['tuning_time']}")
            st.metric("Probe Wait",               f"{m['probe_wait']:.2f}")
            st.metric("Beast Wait",               f"{m['beast_wait']:.2f}")
            st.metric("Schedule Length",          m['schedule_length'])
            st.metric("Index Size (nodes)",       m['index_size'])
            st.metric("Replication Overhead",     m['replication_overhead'])

        with col2:
            st.subheader("Baseline Comparisons")
            st.markdown("**Access-Optimal (no index)**")
            ca, cb = st.columns(2)
            ca.metric("Access Time", f"{m['access_opt_time']:.2f}")
            cb.metric("Tuning Time", f"{m['access_opt_tune']:.2f}")

            st.markdown("**Tune-Optimal (index at start only)**")
            cc, cd = st.columns(2)
            cc.metric("Access Time", f"{m['tune_opt_access']:.2f}")
            cd.metric("Tuning Time", f"{m['tune_opt_tune']}")

        st.subheader("Summary")
        tune_saving = (
            (m['access_opt_tune'] - m['tuning_time']) / m['access_opt_tune'] * 100
            if m['access_opt_tune'] > 0 else 0
        )
        access_gain = (
            (m['tune_opt_access'] - m['access_time']) / m['tune_opt_access'] * 100
            if m['tune_opt_access'] > 0 else 0
        )

        s1, s2 = st.columns(2)
        s1.success(f"Tuning time reduced by **{tune_saving:.1f}%** vs Access-Optimal")
        s2.success(f"Access time improved by **{access_gain:.1f}%** vs Tune-Optimal")

        st.markdown(f"""
        - **r = {r}** replication levels balance probe-wait and beast-wait.
        - Tuning time of **{m['tuning_time']}** probes means the client 
          is active for only ~{m['tuning_time']} buckets per query.
        - The client dozes during the remaining 
          **{m['schedule_length'] - m['tuning_time']}** buckets per cycle.
        """)

        with st.expander("📐 Formulas"):
            st.markdown("**Index tree depth:**")
            st.latex(r"k = \min\{j : n^j \geq D\}")

            st.markdown("**Optimal replication level:**")
            st.latex(
                r"r^* = \left\lfloor"
                r"\frac{\log_n\!\left(D(n-1)+n^{k+1}\right)-1}{2}"
                r"\right\rfloor"
            )

            st.markdown("**Tuning time:**")
            st.latex(r"\text{TuningTime} = k + 1")

            st.markdown("**Access time:**")
            st.latex(
                r"\text{AccessTime} = \underbrace{\tfrac{1}{2}"
                r"\!\left(\frac{I}{q}+\frac{D}{q}\right)}_{\text{ProbeWait}}"
                r"+ \underbrace{\tfrac{1}{2}(I+D)}_{\text{BeastWait}}"
                r"\quad q = n^r,\; I = \text{index size}"
            )


if __name__ == "__main__":
    main()
