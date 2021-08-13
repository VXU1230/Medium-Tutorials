import heapq


def dijkstra(grid):
    seen = set()
    heap = []
    dist = grid[0][0]
    heapq.heappush(heap, (dist, 0, 0))
    row_len = len(grid)
    col_len = len(grid[0])
    min_dist = float('inf')
    while heap:
        dist, row, col = heapq.heappop(heap)
        if row == row_len - 1 and col == col_len - 1:
            min_dist = min(min_dist, dist)
            break
        if (row, col) not in seen:
            seen.add((row, col))
            if row+1 < row_len:
                heapq.heappush(heap, (dist+grid[row+1][col], row+1, col))
            if col+1 < col_len :
                heapq.heappush(heap, (dist+grid[row][col+1], row, col+1))
    return min_dist


if __name__ == "__main__":
    grid = [[1,3,1],[1,5,1],[4,2,1]]
    print("shortest distance: ", dijkstra(grid))

# https://leetcode.com/problems/minimum-path-sum/

