def path_search(row_len, col_len):
    dp = [[0 for _ in range(col_len)] for _ in range(row_len)]
    for r in reversed(range(row_len)):
        for c in reversed(range(col_len)):
            if r == row_len-1 and c == col_len-1:
                dp[r][c] = 1
            else:
                if r+1 < row_len:
                    dp[r][c] += dp[r+1][c]
                if c+1 < col_len:
                    dp[r][c] += dp[r][c+1]
    return dp[0][0]


if __name__ == "__main__":
    m = 3
    n = 7
    print("Number of unique paths: ", path_search(m, n))

# https://leetcode.com/problems/unique-paths/
