from queue import *
import numpy as np


def bfs(rGraph, V, s, t, parent):
    q = Queue()
    visited = np.zeros(V, dtype=bool)
    q.put(s)
    visited[s] = True
    parent[s] = -1

    while not q.empty():
        u = q.get()
        for v in range(V):
            if (not visited[v]) and rGraph[u][v] > 0:
                q.put(v)
                parent[v] = u
                visited[v] = True
    return visited[v]


def dfs(rGraph, V, s, visited):
    stack = [s]
    while stack:
        v = stack.pop()
        if not visited[v]:
            visited[v] = True
            stack.extend([u for u in range(V) if rGraph[v][u]])


def FordFulkerson(graph, s, t):
    print("Running Ford-Fulkerson algorithm")
    rGraph = graph.copy()
    V = len(graph)
    parent = np.zeros(V, dtype="int32")

    while bfs(rGraph, V, s, t, parent):
        pathFlow = float("inf")
        v = t
        while v != s:
            u = parent[v]
            pathFlow = min(pathFlow, rGraph[u][v])
            v = parent[v]

        v = t
        while v != s:
            u = parent[v]
            rGraph[u][v] -= pathFlow
            rGraph[v][u] += pathFlow
            v = parent[v]

    visited = np.zeros(V, dtype=bool)
    dfs(rGraph, V, s, visited)

    cuts = []

    for i in range(V):
        for j in range(V):
            if visited[i] and not visited[j] and graph[i][j]:
                cuts.append((i, j))
    return cuts


def preFlows(C, F, heights, eflows, s):
    # vertices[s,0] = len(vertices)
    heights[s] = len(heights)
    # Height of the source vertex is equal to the total # of vertices

    # edges[s,:,1] = edges[s,:,0]
    F[s, :] = C[s, :]
    # Flow of edges from source is equal to their respective capacities

    for v in range(len(C)):
        # For every vertex v that has an incoming edge from s
        if C[s, v] > 0:
            eflows[v] += C[s, v]
            # Initialize excess flow for v
            C[v, s] = 0
            F[v, s] = -C[s, v]
            # Set capacity of edge from v to s in residual graph to 0


# Returns the first vertex that is not the source and not the sink and
# has a nonzero excess flow
# If non exists return None
def overFlowVertex(vertices, s, t):
    for v in range(len(vertices)):
        if v != s and v != t and vertices[v, 1] > 0:
            return v
    return None


# For a vertex v adjacent to u, we can push if:
#   (1) the flow of the edge u -> v is less than its capacity
#   (2) height of u > height of v
# Flow is the minimum of the remaining possible flow on this edge
# and the excess flow of u
def push(edges, vertices, u):
    for v in range(len(edges[u])):
        if edges[u, v, 1] != edges[u, v, 0]:
            if vertices[u, 0] > vertices[v, 0]:
                flow = min(edges[u, v, 0] - edges[u, v, 1], vertices[u, 1])
                # print "pushing flow", flow, "from", u, "to", v
                vertices[u, 1] -= flow
                vertices[v, 1] += flow
                edges[u, v, 1] += flow
                edges[v, u, 1] -= flow

                return True

    return False


# For a vertex v adjacent to u, we can relabel if
#   (1) the flow of the edge u -> v is less than its capacity
#   (2) the height of v is less than the minimum height
def relabel(edges, vertices, u):
    mh = float("inf")  # Minimum height
    for v in range(len(edges[u])):
        if edges[u, v, 1] != edges[u, v, 0] and vertices[v, 0] < mh:
            mh = vertices[v, 0]
    vertices[u, 0] = mh + 1
    # print "relabeling", u, "with mh", mh + 1


def dfs(rGraph, V, s, visited):

    stack = [s]
    while stack:
        v = stack.pop()
        if not visited[v]:
            visited[v] = True
            stack.extend([u for u in range(V) if rGraph[v][u] > 0])


def pushRelabel(C, s, t):
    print("Running push relabel algorithm")

    def preFlows():
        heights[s] = V
        F[s, :] = C[s, :]
        for v in range(V):
            if C[s, v] > 0:
                excess[v] = C[s, v]
                excess[s] -= C[s, v]
                # C[v,s] = 0
                F[v, s] = -C[s, v]

    def overFlowVertex():
        for v in range(V):
            if v != s and v != t and excess[v] > 0:
                return v
        return None

    def push(u):
        # print "pushing", u
        # assert(excess[u] > 0)
        for v in range(V):
            if C[u, v] > F[u, v] and heights[u] == heights[v] + 1:
                flow = min(C[u, v] - F[u, v], excess[u])
                # if C[u,v] > 0:
                F[u, v] += flow

                # if C[u,v] == 0:
                #     F[v,u] -= flow
                if C[v, u] > F[v, u]:
                    F[v, u] -= flow
                else:
                    F[v, u] = 0
                    C[v, u] = flow
                excess[u] -= flow
                excess[v] += flow
                # F[u,v] += flow
                # F[v,u] -= flow
                return True
        return False

    def relabel(u):
        # assert(excess[u] > 0)
        # print "relabling", u, heights
        assert [heights[u] <= heights[v] for v in range(V) if C[u, v] > F[u, v]]
        heights[u] = 1 + min(
            [heights[v] for v in range(V) if C[u, v] > F[u, v]]
        )

    V = len(C)
    F = np.zeros((V, V))
    heights = np.zeros(V)
    excess = np.zeros(V)

    preFlows()

    while True:
        u = overFlowVertex()
        # print "overflowing vertex is", u
        if u == None:
            break
        if not push(u):
            relabel(u)
    # Max flow is equal to the excess flow of the sink
    # return vertices[t,1]
    print("Max flow", excess[t])
    # print C
    # print F
    # print C-F
    # print heights
    # print excess

    visited = np.zeros(V, dtype=bool)
    dfs(C - F, V, s, visited)

    cuts = []

    for u in range(V):
        for v in range(V):
            if visited[u] and not visited[v] and C[u, v]:
                cuts.append((u, v))
    return cuts