import collections


class Node:
    def __init__(self, val=None):
        self.val = val
        self.left = None
        self.right = None


def get_root():
    values = iter([1, 6, 4, 7, None, None, 9, 8, None, None,
              10, None, None, 5, None, None, 2, 3, None, None, 11, None, None])

    def tree_recur(itr):
        val = next(itr)
        if val is not None:
            node = Node(val)
            node.left = tree_recur(itr)
            node.right = tree_recur(itr)
            return node

    return tree_recur(values)


def dfs():
    root = get_root()
    res = float("inf")

    def dfs_search(node, depth):
        if node is not None:
            val = node.val
            print(val, end=" ")
            if val >= 10:
                nonlocal res
                res = min(res, depth)
            else:
                dfs_search(node.left, depth+1)
                dfs_search(node.right, depth + 1)

    dfs_search(root, 0)
    if res < float("inf"):
        return res
    return -1


def bfs():
    root = get_root()
    queue = collections.deque()
    queue.appendleft((root, 0))
    res = -1

    while queue:
        node, depth = queue.pop()
        print(node.val, end=" ")
        if node.val >= 10:
            res = depth
            break

        if node.left:
            queue.appendleft((node.left, depth+1))
        if node.right:
            queue.appendleft((node.right, depth+1))
    return res


def iddfs():
    root = get_root()
    res = float("inf")

    def iddfs_search(node, depth, limit):
        if depth <= limit and node is not None:
            val = node.val
            print(val, end=" ")
            if val >= 10:
                nonlocal res
                res = min(res, depth)
            else:
                iddfs_search(node.left, depth + 1, limit)
                iddfs_search(node.right, depth + 1, limit)

    for limit in range(1, 5):
        print("\nmax depth: ", limit)
        iddfs_search(root, 0, limit)
        if res < float("inf"):
            return res
    return -1


if __name__ == "__main__":
    print("\nBFS")
    print("\nshortest depth: ", bfs())
    print("\nDFS")
    print("\nshortest depth: ", dfs())
    print("\nIDDFS", end="")
    print("\nshortest depth: ", iddfs())