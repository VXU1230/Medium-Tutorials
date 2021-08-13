import heapq


def dijkstra(grid):
    if grid[0][0] == 1 or grid[-1][-1] == 1:
        return -1
    n = len(grid)
    heap = []

    seen = set()
    heapq.heappush(heap, (1, 1, 0, 0))
    cache = {(0, 0): 1}
    while heap:
        (cost, dist, x, y) = heapq.heappop(heap)
        print("({},{})".format(x, y), end=" ")
        if x == n - 1 and y == n - 1:
            return dist
        if (x,y) not in seen:
            seen.add((x, y))
            for (nei_x, nei_y) in get_nei(x, y):
                if 0 <= nei_x < n and 0 <= nei_y < n and grid[nei_x][nei_y] == 0:
                    new_dist = dist + 1
                    if new_dist < cache.get((nei_x, nei_y), float('inf')):
                        cache[(nei_x, nei_y)] = new_dist
                        heapq.heappush(heap, (new_dist, new_dist, nei_x, nei_y))
    return -1


def a_star(grid):
    def get_heuristic(current, target):
        return max(abs(current[0] - target[0]), abs(current[1] - target[1]))

    if grid[0][0] == 1 or grid[-1][-1] == 1:
        return -1

    n = len(grid)
    target = (n - 1, n - 1)
    heap = []
    seen = set()
    heuristic = get_heuristic((0, 0), target)
    heapq.heappush(heap, (heuristic + 1, 1, 0, 0))
    cache = {(0, 0): heuristic + 1}
    while heap:
        (cost, dist, x, y) = heapq.heappop(heap)
        print("({},{})".format(x, y), end=" ")
        if x == n - 1 and y == n - 1:
            return dist
        if (x, y) not in seen:
            seen.add((x, y))
            for (nei_x, nei_y) in get_nei(x, y):
                if 0 <= nei_x < n and 0 <= nei_y < n and grid[nei_x][nei_y] == 0:
                    new_dist = dist + 1
                    heuristic = get_heuristic((nei_x, nei_y), target)
                    if new_dist + heuristic < cache.get((nei_x, nei_y), float('inf')):
                        cache[(nei_x, nei_y)] = new_dist + heuristic
                        heapq.heappush(heap, (new_dist + heuristic, new_dist, nei_x, nei_y))

    return -1


def get_nei(x, y):
    return [(x + 1, y), (x - 1, y), (x + 1, y - 1), (x - 1, y - 1), (x + 1, y + 1), (x - 1, y + 1), (x, y + 1),
            (x, y - 1)]


if __name__ == "__main__":
    grid = [[0,0,0,0,1,1,1,1,0],
            [0,1,1,0,0,0,0,1,0],
            [0,0,1,0,0,0,0,0,0],
            [1,1,0,0,1,0,0,1,1],
            [0,0,1,1,1,0,1,0,1],
            [0,1,0,1,0,0,0,0,0],
            [0,0,0,1,0,1,0,0,0],
            [0,1,0,1,1,0,0,0,0],
            [0,0,0,0,0,1,0,1,0]]

    print("Dijkstra path: ")
    dist_d = dijkstra(grid)
    print("\nshortest distance: {}\n".format(dist_d))
    print("A* path: ")
    dist_a = a_star(grid)
    print("\nshortest distance: {}".format(dist_a))

# https://leetcode.com/problems/shortest-path-in-binary-matrix/