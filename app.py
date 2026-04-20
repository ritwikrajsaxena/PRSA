import streamlit as st
import math
import graphviz
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional
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
    children_ids: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    data_bucket_ids: List[int] = field(default_factory=list)
    min_key: int = 0
    max_key: int = 0

    def is_leaf(self) -> bool:
        return len(self.children_ids) == 0

    def last_key(self) -> int:
        return max(self.keys) if self.keys else 0


@dataclass
class DataBucket:
    """Represents a data bucket in the broadcast"""
    bucket_id: int
    keys: List[int] = field(default_factory=list)


@dataclass
class AccessStep:
    """Represents a step in the access protocol"""
    step_num: int
    action: str
    position: int
    bucket_accessed: str
    mode: str
    details: str


# ============================================================================
# INDEX TREE CONSTRUCTION
# ============================================================================

class IndexTree:
    def __init__(
        self,
        num_data_buckets: int,
        bucket_capacity: int,
        key_values: Optional[List[int]] = None
    ):
        self.num_data_buckets = num_data_buckets
        self.bucket_capacity = max(2, bucket_capacity)
        self.num_levels = self._calculate_levels()
        self.root: Optional[IndexNode] = None
        self.data_buckets: List[DataBucket] = []
        self.all_nodes: Dict[str, IndexNode] = {}
        self.nodes_by_level: Dict[int, List[str]] = {}

        if key_values is None:
            self.key_values = list(range(1, num_data_buckets + 1))
        else:
            self.key_values = sorted(key_values)[:num_data_buckets]
            if len(self.key_values) < num_data_buckets:
                next_key = (max(self.key_values) + 1) if self.key_values else 1
                while len(self.key_values) < num_data_buckets:
                    self.key_values.append(next_key)
                    next_key += 1

        self._build_tree()

    def _calculate_levels(self) -> int:
        """k = ⌈log_n(Data)⌉"""
        if self.num_data_buckets <= 1:
            return 1
        n = self.bucket_capacity
        levels = 0
        capacity = 1
        while capacity < self.num_data_buckets:
            capacity *= n
            levels += 1
        return max(1, levels)

    def _build_tree(self):
        """Build index tree bottom-up."""
        n = self.bucket_capacity

        self.data_buckets = [
            DataBucket(bucket_id=i, keys=[self.key_values[i]])
            for i in range(self.num_data_buckets)
        ]

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

                for child_id in chunk_ids:
                    self.all_nodes[child_id].parent_id = node_id

                self.all_nodes[node_id] = node
                self.nodes_by_level[current_level].append(node_id)
                next_level_ids.append(node_id)

            current_level_ids = next_level_ids
            current_level -= 1

        if current_level_ids:
            root_candidate_id = current_level_ids[0]
            root_node = self.all_nodes[root_candidate_id]

            del self.all_nodes[root_candidate_id]
            old_level = root_node.level

            for child_id in root_node.children_ids:
                self.all_nodes[child_id].parent_id = "ROOT"

            root_node.node_id = "ROOT"
            root_node.level = 0
            self.all_nodes["ROOT"] = root_node
            self.root = root_node

            if old_level in self.nodes_by_level:
                level_list = self.nodes_by_level[old_level]
                if root_candidate_id in level_list:
                    level_list.remove(root_candidate_id)
            self.nodes_by_level.setdefault(0, []).append("ROOT")

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
        path = []
        current = self.get_parent(node)
        while current is not None:
            path.append(current)
            current = self.get_parent(current)
        return list(reversed(path))

    def find_lca(self, node1: IndexNode, node2: IndexNode) -> Optional[IndexNode]:
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

    def count_nodes_at_level(self, level: int) -> int:
        return len(self.nodes_by_level.get(level, []))

    def get_total_index_nodes(self) -> int:
        return len(self.all_nodes)


# ============================================================================
# DISTRIBUTED INDEXING ALGORITHM
# ============================================================================

class DistributedIndex:
    def __init__(self, index_tree: IndexTree, replication_level: int):
        self.tree = index_tree
        self.r = max(0, min(replication_level, index_tree.num_levels - 1))
        self.k = index_tree.num_levels
        self.n = index_tree.bucket_capacity
        self.nrr: List[IndexNode] = []
        self.broadcast_schedule: List[Dict] = []

        self._identify_nrr()
        self._generate_schedule()

    def _identify_nrr(self):
        """NRR nodes at level r (paper: level r+1 in 1-indexed)"""
        nrr_level = min(self.r, self.k - 1)
        self.nrr = self.tree.get_nodes_at_level(nrr_level)

        if not self.nrr:
            for level in range(self.k - 1, -1, -1):
                nodes = self.tree.get_nodes_at_level(level)
                if nodes:
                    self.nrr = nodes
                    break

    def _generate_schedule(self):
        """Generate broadcast schedule: ⟨Rep(B), Ind(B), Data(B)⟩ per NRR"""
        self.broadcast_schedule = []
        position = 0

        for i, B in enumerate(self.nrr):
            # Rep(B)
            if i == 0:
                rep_nodes = self.tree.get_path_to_root(B)
            else:
                B_prev = self.nrr[i - 1]
                lca = self.tree.find_lca(B_prev, B)
                rep_nodes = self.tree.get_path_between(lca, B) if lca else []

            for node in rep_nodes:
                self.broadcast_schedule.append({
                    'type': 'rep',
                    'node': node,
                    'section': i,
                    'position': position,
                })
                position += 1

            # Ind(B) - BFS traversal
            bfs_queue = deque([B])
            while bfs_queue:
                node = bfs_queue.popleft()
                self.broadcast_schedule.append({
                    'type': 'ind',
                    'node': node,
                    'section': i,
                    'position': position,
                })
                position += 1
                bfs_queue.extend(self.tree.get_children(node))

            # Data(B)
            for data_id in sorted(self._collect_data_ids(B)):
                if data_id < len(self.tree.data_buckets):
                    db = self.tree.data_buckets[data_id]
                    self.broadcast_schedule.append({
                        'type': 'data',
                        'bucket': db,
                        'section': i,
                        'position': position,
                    })
                    position += 1

    def _collect_data_ids(self, node: IndexNode) -> List[int]:
        ids = []
        stack = [node]
        while stack:
            n = stack.pop()
            ids.extend(n.data_bucket_ids)
            stack.extend(self.tree.get_children(n))
        return ids

    @staticmethod
    def calculate_optimal_r(D: int, n: int, k: int) -> int:
        """
        Paper formula (Section 4.3):
        r* = ⌊½ × (log_n((D(n-1) + n^(k+1)) / (n-1)) - 1)⌋ + 1
        """
        if k <= 1 or n <= 1:
            return 0

        try:
            numerator = D * (n - 1) + n**(k + 1)
            denominator = n - 1
            inner = numerator / denominator

            if inner <= 0:
                return 0

            log_val = math.log(inner, n)
            r_star = math.floor(0.5 * (log_val - 1)) + 1

            return max(0, min(k - 1, r_star))
        except (ValueError, ZeroDivisionError, OverflowError):
            return 0


# ============================================================================
# ACCESS SIMULATOR
# ============================================================================

class AccessSimulator:
    def __init__(self, distributed_index: DistributedIndex):
        self.di = distributed_index
        self.schedule = distributed_index.broadcast_schedule
        self.tree = distributed_index.tree

    def _node_contains_key(self, node: IndexNode, key: int) -> bool:
        return node.min_key <= key <= node.max_key

    def simulate_access(
        self, tune_in_position: int, target_key: int
    ) -> List[AccessStep]:
        """
        Paper access protocol (Section 4.2):
        1. Initial probe - get offset to control index
        2. Read control index
        3. Follow index pointers (up to k probes)
        4. Download data
        """
        steps: List[AccessStep] = []
        sched_len = max(1, len(self.schedule))
        pos = tune_in_position % sched_len
        step_num = 1
        last_active_pos = pos

        # Step 1: Initial probe
        steps.append(AccessStep(
            step_num=step_num,
            action="Initial Probe",
            position=pos,
            bucket_accessed=self._bucket_name(pos),
            mode="active",
            details=f"Get offset to next control index (position {pos})"
        ))
        step_num += 1
        pos += 1

        found = False
        while pos < sched_len:
            seg = self.schedule[pos]

            if seg['type'] == 'data':
                bucket: DataBucket = seg['bucket']
                if target_key in bucket.keys:
                    if pos > last_active_pos + 1:
                        steps.append(AccessStep(
                            step_num=step_num,
                            action="Doze Mode",
                            position=last_active_pos + 1,
                            bucket_accessed="-",
                            mode="doze",
                            details=f"Sleep {pos - last_active_pos - 1} buckets"
                        ))
                        step_num += 1

                    steps.append(AccessStep(
                        step_num=step_num,
                        action="Download Data",
                        position=pos,
                        bucket_accessed=f"Data Bucket {bucket.bucket_id}",
                        mode="active",
                        details=f"✅ Found key {target_key}"
                    ))
                    found = True
                    break

            elif seg['type'] in ('rep', 'ind'):
                node: IndexNode = seg['node']
                if self._node_contains_key(node, target_key):
                    if pos > last_active_pos + 1:
                        steps.append(AccessStep(
                            step_num=step_num,
                            action="Doze Mode",
                            position=last_active_pos + 1,
                            bucket_accessed="-",
                            mode="doze",
                            details=f"Sleep {pos - last_active_pos - 1} buckets"
                        ))
                        step_num += 1

                    node_type = "Control Index" if seg['type'] == 'rep' else "Index"
                    steps.append(AccessStep(
                        step_num=step_num,
                        action=f"Probe {node_type}",
                        position=pos,
                        bucket_accessed=node.node_id,
                        mode="active",
                        details=f"Range [{node.min_key}, {node.max_key}] contains {target_key}"
                    ))
                    step_num += 1
                    last_active_pos = pos

            pos += 1

        if not found:
            steps.append(AccessStep(
                step_num=step_num,
                action="Wait for Next Beast",
                position=sched_len,
                bucket_accessed="-",
                mode="doze",
                details=f"Key {target_key} not found, wait for next broadcast cycle"
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
# METRICS CALCULATION (PAPER-ACCURATE)
# ============================================================================

def calculate_metrics(tree: IndexTree, di: DistributedIndex) -> Dict:
    """
    EXACT formulas from paper Section 4.3:
    
    ProbeWait  = ½ × (Ind(B) + Data(B))
               = ½ × ((n^(k-r) - 1)/(n-1) + Data/n^r)
               
    BeastWait  = ½ × ((n^r - 1) + Index + Data)
    
    AccessTime = ProbeWait + BeastWait
    
    TuningTime = ⌈log_n(Data)⌉ + 3 = k + 3
    
    IndexOverhead = n^r - 1
    """
    n = tree.bucket_capacity
    k = tree.num_levels
    r = di.r
    D = tree.num_data_buckets

    # Index size (actual nodes in tree)
    index_size = tree.get_total_index_nodes()

    # Theoretical index size for balanced tree
    # Index = (n^k - 1) / (n - 1)
    if n > 1:
        theoretical_index = (n**k - 1) // (n - 1)
    else:
        theoretical_index = k

    # Number of NRR sections = n^r
    q = n ** r

    # Average Ind(B) = (n^(k-r) - 1) / (n - 1)
    if n > 1 and k > r:
        ind_B = (n**(k - r) - 1) / (n - 1)
    else:
        ind_B = 1

    # Average Data(B) = Data / n^r
    data_B = D / q

    # =========================================================================
    # PAPER FORMULAS (Section 4.3)
    # =========================================================================

    # Probe Wait = ½ × (Ind(B) + Data(B))
    probe_wait = 0.5 * (ind_B + data_B)

    # Index Overhead = n^r - 1 (extra replicated buckets)
    index_overhead = (n**r - 1) if r > 0 else 0

    # Total broadcast length = Index + Data + IndexOverhead
    total_broadcast = index_size + D + index_overhead

    # Beast Wait = ½ × total broadcast length
    beast_wait = 0.5 * total_broadcast

    # Access Time = Probe Wait + Beast Wait
    access_time = probe_wait + beast_wait

    # Tuning Time = k + 3 (paper Section 4.3)
    # 1 (initial probe) + 1 (control index) + k (index levels) + 1 (data)
    tuning_time = k + 3

    # =========================================================================
    # BASELINES (Paper Section 2)
    # =========================================================================

    # Access-Optimal: no index, scan half the data
    access_opt_access = D / 2.0
    access_opt_tune = D / 2.0

    # Tune-Optimal: index at start, must wait full cycle if missed
    # "access time is (Data + Index)" - paper says this is the FULL cycle
    tune_opt_access = D + index_size
    tune_opt_tune = k + 1  # Just follow index pointers + 1 data read

    # =========================================================================
    # (1,m) INDEXING (Paper Section 3)
    # =========================================================================

    # Optimal m* = sqrt(Data / Index)
    if index_size > 0:
        m_star = max(1, int(math.sqrt(D / index_size)))
    else:
        m_star = 1

    # Access time = ½ × ((m+1) × Index + (1 + 1/m) × Data)
    one_m_access = 0.5 * ((m_star + 1) * index_size + (1 + 1/m_star) * D)

    # Tuning time = 2 + ⌈log_n(Data)⌉ = k + 2
    one_m_tune = k + 2

    return {
        # Core metrics
        'access_time': round(access_time, 2),
        'tuning_time': tuning_time,
        'probe_wait': round(probe_wait, 2),
        'beast_wait': round(beast_wait, 2),

        # Section breakdown
        'ind_per_section': round(ind_B, 2),
        'data_per_section': round(data_B, 2),
        'nrr_count': q,

        # Broadcast info
        'schedule_length': len(di.broadcast_schedule),
        'total_broadcast': total_broadcast,
        'index_size': index_size,
        'index_overhead': index_overhead,

        # Baselines
        'access_opt_access': access_opt_access,
        'access_opt_tune': access_opt_tune,
        'tune_opt_access': tune_opt_access,
        'tune_opt_tune': tune_opt_tune,

        # (1,m) indexing
        'one_m_optimal_m': m_star,
        'one_m_access': round(one_m_access, 2),
        'one_m_tune': one_m_tune,
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_tree_visualization(tree: IndexTree, r: int) -> graphviz.Digraph:
    dot = graphviz.Digraph(comment='Index Tree')
    dot.attr(rankdir='TB')

    def add_node(node: IndexNode):
        is_replicated = node.level < r
        is_nrr = node.level == r
        fill = 'lightblue' if is_replicated else ('gold' if is_nrr else 'lightgreen')
        border = 'red' if is_nrr else 'black'
        label = f"{node.node_id}\n[{node.min_key}–{node.max_key}]"
        dot.node(node.node_id, label, fillcolor=fill, style='filled',
                 color=border, penwidth='2' if is_nrr else '1')
        for child in tree.get_children(node):
            add_node(child)
            dot.edge(node.node_id, child.node_id)

    if tree.root:
        add_node(tree.root)

    dot.attr('node', shape='ellipse', style='filled', fillcolor='moccasin')
    for db in tree.data_buckets:
        dot.node(f"D{db.bucket_id}", f"D{db.bucket_id}\n{db.keys}")

    for nid in tree.all_nodes:
        node = tree.all_nodes[nid]
        for did in node.data_bucket_ids:
            dot.edge(node.node_id, f"D{did}", style='dashed', color='gray')

    return dot


def create_schedule_dataframe(schedule: List[Dict]) -> pd.DataFrame:
    rows = []
    for seg in schedule:
        if seg['type'] == 'data':
            db = seg['bucket']
            rows.append({
                'Pos': seg['position'],
                'Type': 'DATA',
                'Name': f"D{db.bucket_id}",
                'Keys': str(db.keys),
                'Section': seg['section'],
            })
        else:
            node = seg['node']
            rows.append({
                'Pos': seg['position'],
                'Type': seg['type'].upper(),
                'Name': node.node_id,
                'Keys': f"[{node.min_key}-{node.max_key}]",
                'Section': seg['section'],
            })
    return pd.DataFrame(rows)


# ============================================================================
# STREAMLIT APP
# ============================================================================

@st.cache_resource
def build_index_cached(D: int, n: int, keys: tuple, r: int):
    tree = IndexTree(D, n, list(keys))
    di = DistributedIndex(tree, r)
    return tree, di


def main():
    st.set_page_config(page_title="Distributed Indexing on Air", layout="wide")
    st.title("📡 Distributed Indexing on Air")
    st.caption("Based on: Imielinski, Viswanathan & Badrinath, SIGMOD 1994")

    # Sidebar
    st.sidebar.header("Parameters")
    D = st.sidebar.number_input("Data Buckets (D)", 1, 10000, 27)
    n = st.sidebar.number_input("Bucket Capacity (n)", 2, 100, 3)
    
    keys = list(range(1, D + 1))
    tmp_tree = IndexTree(D, n, keys)
    k = tmp_tree.num_levels
    
    optimal_r = DistributedIndex.calculate_optimal_r(D, n, k)
    st.sidebar.info(f"k = {k} levels | Optimal r* = {optimal_r}")
    
    r = st.sidebar.slider("Replication Level (r)", 0, max(0, k-1), optimal_r)
    
    tree, di = build_index_cached(D, n, tuple(keys), r)
    
    tune_pos = st.sidebar.number_input("Tune-in Position", 0, 
                                        max(0, len(di.broadcast_schedule)-1), 0)
    target_key = st.sidebar.selectbox("Target Key", keys, index=len(keys)//2)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🌳 Tree", "📋 Schedule", "🔍 Simulation", "📊 Metrics"])

    with tab1:
        st.header("Index Tree Structure")
        try:
            dot = create_tree_visualization(tree, r)
            st.graphviz_chart(dot)
        except Exception as e:
            st.error(f"Visualization error: {e}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **Tree Properties:**
            - Levels (k): **{k}**
            - Fanout (n): **{n}**
            - Replication (r): **{r}**
            - Data Buckets: **{D}**
            - Index Nodes: **{len(tree.all_nodes)}**
            - NRR Count: **{len(di.nrr)}**
            """)
        with col2:
            st.markdown("""
            **Legend:**
            - 🔵 Blue: Replicated (levels 0..r-1)
            - 🟡 Gold: NRR nodes (level r)
            - 🟢 Green: Non-replicated
            - 🟠 Moccasin: Data buckets
            """)

    with tab2:
        st.header("Broadcast Schedule")
        df = create_schedule_dataframe(di.broadcast_schedule)
        st.dataframe(df, use_container_width=True, height=400)
        
        st.subheader("Schedule Structure")
        st.markdown(f"""
        **Total length:** {len(di.broadcast_schedule)} buckets
        
        Each section = Rep(B) + Ind(B) + Data(B)
        - {len(di.nrr)} NRR sections
        """)

    with tab3:
        st.header("Access Protocol Simulation")
        
        if st.button("▶ Run Simulation", type="primary"):
            sim = AccessSimulator(di)
            st.session_state['steps'] = sim.simulate_access(tune_pos, target_key)
        
        if 'steps' in st.session_state:
            steps = st.session_state['steps']
            active = sum(1 for s in steps if s.mode == 'active')
            
            st.metric("Active Probes", active, 
                      help=f"Paper predicts ≤ k+3 = {k+3} probes")
            
            for step in steps:
                icon = "🔋" if step.mode == 'active' else "😴"
                with st.expander(f"Step {step.step_num}: {step.action} {icon}"):
                    st.markdown(f"""
                    - **Position:** {step.position}
                    - **Bucket:** {step.bucket_accessed}
                    - **Mode:** {step.mode.upper()}
                    - **Details:** {step.details}
                    """)

    with tab4:
        st.header("Performance Metrics (Paper Formulas)")
        m = calculate_metrics(tree, di)

        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Distributed Indexing")
            st.metric("Access Time", f"{m['access_time']} buckets")
            st.metric("Tuning Time", f"{m['tuning_time']} probes")
            st.metric("Probe Wait", f"{m['probe_wait']} buckets")
            st.metric("Beast Wait", f"{m['beast_wait']} buckets")
            st.metric("Index Overhead", f"{m['index_overhead']} buckets")

        with col2:
            st.subheader("Access-Optimal")
            st.caption("No index, sequential scan")
            st.metric("Access Time", f"{m['access_opt_access']} buckets")
            st.metric("Tuning Time", f"{m['access_opt_tune']} buckets")

        with col3:
            st.subheader("Tune-Optimal")
            st.caption("Index once at beast start")
            st.metric("Access Time", f"{m['tune_opt_access']} buckets")
            st.metric("Tuning Time", f"{m['tune_opt_tune']} probes")

        st.markdown("---")
        st.subheader("(1,m) Indexing Comparison")
        st.markdown(f"""
        - Optimal m* = **{m['one_m_optimal_m']}**
        - Access Time = **{m['one_m_access']}** buckets
        - Tuning Time = **{m['one_m_tune']}** probes
        """)

        st.markdown("---")
        st.subheader("Paper Formulas")
        st.latex(r"\text{ProbeWait} = \frac{1}{2} \times \left( \frac{n^{k-r}-1}{n-1} + \frac{D}{n^r} \right)")
        st.latex(r"\text{BeastWait} = \frac{1}{2} \times \left( (n^r - 1) + I + D \right)")
        st.latex(r"\text{AccessTime} = \text{ProbeWait} + \text{BeastWait}")
        st.latex(r"\text{TuningTime} = k + 3")
        st.latex(r"r^* = \left\lfloor \frac{1}{2} \left( \log_n \frac{D(n-1) + n^{k+1}}{n-1} - 1 \right) \right\rfloor + 1")


if __name__ == "__main__":
    main()
