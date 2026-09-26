"""Hybrid REL + DSR routing simulator for the CardioLens-IoT sensor network.

A lightweight, dependency-free reference implementation of the routing layer:

* REL (routing by energy and link quality): proactive routes to frequently used destinations
  (the cloud gateway). The next hop is chosen by link quality and residual energy.
* DSR (dynamic source routing): reactive route discovery with a route cache, used for
  infrequent destinations.
* Hybrid: REL for frequent destinations, DSR for everything else.

The simulator compares DSR-only, REL-only and Hybrid on the same random topology and traffic.
It is an illustrative model with simplified radio and energy assumptions, not a
hardware measurement.

    python iot/network/hybrid_routing.py --nodes 40 --packets 3000 --seed 7
"""
from __future__ import annotations

import argparse
import heapq
import math
import random
from collections import deque
from dataclasses import dataclass

# --- simplified radio / energy model (units: mJ) ---------------------------------------------
E_INIT = 2000.0         # initial energy of a battery-powered node
E_ELEC = 0.05           # electronics cost per packet (tx or rx)
E_AMP = 0.0001          # amplifier cost per packet per m^2
E_CTRL = 0.02           # cost of handling one control message (RREQ / RREP / beacon)
MAX_RETRIES = 3         # link-layer retransmissions per hop
RADIO_RANGE = 30.0      # m


@dataclass
class Node:
    nid: int
    x: float
    y: float
    energy: float = E_INIT
    is_gateway: bool = False    # mains-powered, unlimited energy

    @property
    def alive(self) -> bool:
        return self.is_gateway or self.energy > 0


class Network:
    def __init__(self, n_nodes: int, area: float, seed: int):
        self.rng = random.Random(seed)
        while True:
            self.nodes = [Node(i, self.rng.uniform(0, area), self.rng.uniform(0, area)) for i in range(n_nodes)]
            self.gateway = 0
            self.nodes[0].x, self.nodes[0].y, self.nodes[0].is_gateway = area / 2, area / 2, True
            self.adj = {i: {} for i in range(n_nodes)}
            for a in self.nodes:
                for b in self.nodes:
                    d = math.dist((a.x, a.y), (b.x, b.y))
                    if a.nid < b.nid and d <= RADIO_RANGE:
                        # link quality (0-1): decays with distance, with per-link variation (LQI-like)
                        q = max(0.35, min(0.99, 1.0 - 0.6 * (d / RADIO_RANGE) ** 2 + self.rng.uniform(-0.1, 0.1)))
                        self.adj[a.nid][b.nid] = self.adj[b.nid][a.nid] = (d, q)
            if self._connected():
                break

    def _connected(self) -> bool:
        seen, todo = {0}, [0]
        while todo:
            for v in self.adj[todo.pop()]:
                if v not in seen:
                    seen.add(v); todo.append(v)
        return len(seen) == len(self.nodes)

    def neighbours(self, u: int):
        return [v for v in self.adj[u] if self.nodes[v].alive]

    def spend(self, nid: int, mj: float) -> None:
        if not self.nodes[nid].is_gateway:
            self.nodes[nid].energy -= mj


class Stats:
    def __init__(self):
        self.sent = self.delivered = self.hops = self.retx = self.ctrl = 0
        self.first_death = None

    def row(self, net: Network, name: str) -> dict:
        sensors = [n for n in net.nodes if not n.is_gateway]
        used = sum(E_INIT - max(n.energy, 0) for n in sensors)
        return {"policy": name,
                "delivery_%": 100 * self.delivered / max(self.sent, 1),
                "avg_hops": self.hops / max(self.delivered, 1),
                "retx/pkt": self.retx / max(self.sent, 1),
                "ctrl_msgs": self.ctrl,
                "energy_J": used / 1000,
                "min_residual_%": 100 * min(max(n.energy, 0) for n in sensors) / E_INIT,
                "first_death_pkt": self.first_death if self.first_death is not None else "none"}


class Router:
    """Implements REL, DSR and the hybrid policy on top of a Network."""

    def __init__(self, net: Network, policy: str, frequent: set, refresh: int = 200,
                 w_link: float = 3.0, w_energy: float = 2.0):
        self.net, self.policy, self.frequent = net, policy, frequent
        self.refresh, self.w_link, self.w_energy = refresh, w_link, w_energy
        self.cache: dict = {}          # DSR route cache: (src, dst) -> path
        self.rel_next: dict = {}       # REL table: dst -> {node: next_hop}
        self.stats = Stats()
        self.pkt = 0
        # destinations whose routes are maintained proactively (periodic beacons)
        if policy == "REL":
            self.proactive = set(range(len(net.nodes)))    # a purely proactive protocol tracks every node
        elif policy == "Hybrid":
            self.proactive = set(frequent)                  # hybrid: only the frequently used destinations
        else:
            self.proactive = set()

    # ---- REL: proactive, energy- and link-quality-aware ------------------------------------
    def _rel_cost(self, u: int, v: int) -> float:
        _, q = self.net.adj[u][v]
        e = 1.0 if self.net.nodes[v].is_gateway else self.net.nodes[v].energy / E_INIT
        return 1.0 + self.w_link * (1 - q) + self.w_energy * (1 - e)

    def _rel_build(self, dst: int) -> None:
        """Dijkstra from the destination: every node learns its best next hop towards dst."""
        dist, nxt, pq = {dst: 0.0}, {}, [(0.0, dst)]
        while pq:
            d, v = heapq.heappop(pq)
            if d > dist.get(v, math.inf):
                continue
            for u in self.net.neighbours(v):
                nd = d + self._rel_cost(u, v)
                if nd < dist.get(u, math.inf):
                    dist[u], nxt[u] = nd, v
                    heapq.heappush(pq, (nd, u))
        self.rel_next[dst] = nxt
        for n in self.net.nodes:                       # one beacon per live node per refresh
            if n.alive:
                self.net.spend(n.nid, E_CTRL); self.stats.ctrl += 1

    def _maintain(self) -> None:
        """Periodic proactive maintenance: refresh the REL table of every tracked destination."""
        if (self.pkt - 1) % self.refresh == 0:
            for dst in self.proactive:
                self._rel_build(dst)

    def _rel_path(self, src: int, dst: int):
        if dst not in self.rel_next:
            self._rel_build(dst)
        path, u = [src], src
        while u != dst:
            u = self.rel_next[dst].get(u)
            if u is None or u in path:
                return None
            path.append(u)
        return path

    # ---- DSR: reactive route discovery with a route cache -----------------------------------
    def _dsr_discover(self, src: int, dst: int):
        """Flood a route request (each node rebroadcasts once); the reply returns the first path found."""
        parent, q = {src: None}, deque([src])
        while q:
            u = q.popleft()
            self.net.spend(u, E_CTRL); self.stats.ctrl += 1          # RREQ broadcast
            if u == dst:
                break
            for v in self.net.neighbours(u):
                if v not in parent:
                    parent[v] = u; q.append(v)
        if dst not in parent:
            return None
        path, v = [], dst
        while v is not None:
            path.append(v); v = parent[v]
        path.reverse()
        for n in path:                                            # RREP travels back along the path
            self.net.spend(n, E_CTRL); self.stats.ctrl += 1
        return path

    def _dsr_path(self, src: int, dst: int):
        path = self.cache.get((src, dst))
        if path is None or not all(self.net.nodes[n].alive for n in path):
            path = self._dsr_discover(src, dst)
            if path:
                self.cache[(src, dst)] = path
        return path

    # ---- forwarding ---------------------------------------------------------------------------
    def send(self, src: int, dst: int) -> None:
        self.pkt += 1
        self.stats.sent += 1
        self._maintain()
        use_rel = self.policy == "REL" or (self.policy == "Hybrid" and dst in self.frequent)
        path = self._rel_path(src, dst) if use_rel else self._dsr_path(src, dst)
        if not path:
            return
        for u, v in zip(path, path[1:]):
            d, q = self.net.adj[u][v]
            for attempt in range(MAX_RETRIES + 1):
                self.net.spend(u, E_ELEC + E_AMP * d * d)
                self.net.spend(v, E_ELEC)
                if self.net.rng.random() < q:
                    break
                self.stats.retx += 1
            else:
                self.cache.pop((src, dst), None)               # link failed: route error
                return
        self.stats.delivered += 1
        self.stats.hops += len(path) - 1
        if self.stats.first_death is None and any(not n.alive for n in self.net.nodes):
            self.stats.first_death = self.pkt


def simulate(policy: str, n_nodes: int, area: float, packets: int, seed: int, p_gateway: float) -> dict:
    net = Network(n_nodes, area, seed)
    router = Router(net, policy, frequent={net.gateway})
    traffic = random.Random(seed + 1)
    sensors = list(range(1, n_nodes))
    for _ in range(packets):
        src = traffic.choice(sensors)
        if not net.nodes[src].alive:
            router.stats.sent += 1
            continue
        dst = net.gateway if traffic.random() < p_gateway else traffic.choice([s for s in sensors if s != src])
        router.send(src, dst)
    return router.stats.row(net, policy)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare DSR, REL and hybrid routing for ECG uploads.")
    ap.add_argument("--nodes", type=int, default=40)
    ap.add_argument("--area", type=float, default=100.0, help="side of the square area in metres")
    ap.add_argument("--packets", type=int, default=3000)
    ap.add_argument("--p-gateway", type=float, default=0.9, help="share of packets addressed to the gateway")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    rows = [simulate(p, a.nodes, a.area, a.packets, a.seed, a.p_gateway) for p in ("DSR", "REL", "Hybrid")]
    cols = list(rows[0])
    print(f"\nSimulation: {a.nodes} nodes, {a.area:.0f} m x {a.area:.0f} m, {a.packets} packets, "
          f"{a.p_gateway:.0%} to gateway, seed {a.seed}\n")
    print(" | ".join(f"{c:>15}" for c in cols))
    for r in rows:
        print(" | ".join(f"{r[c]:>15.2f}" if isinstance(r[c], float) else f"{str(r[c]):>15}" for c in cols))


if __name__ == "__main__":
    main()
