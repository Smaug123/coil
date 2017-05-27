#!/usr/bin/env python3

import copy
import itertools
import board


def str_row(row):
    return ''.join(str(c) for c in row)

def thin_array(arr):
    """
    Given a list of lists of 0/1, thin out so that the 1's are in a line.
    """
    # Algorithm: throw away any 1 which is at the corner of a 2x2 square of 1s.
    changed = True
    while changed:
        changed = False
        for r, row in enumerate(arr):
            if r == 0 or r == len(arr) - 1:
                continue
            for c, cell in enumerate(row):
                # If at the top-left corner of a 2x2 square of 1s:
                if c == 0 or c == len(row) - 1:
                    continue
                if cell == 0:
                    continue
                if row[c-1] == 0 and row[c+1] == 1 and cell == 1:
                    if arr[r-1][c] == 0 and arr[r+1][c] == 1:
                        if arr[r+1][c+1] == 1:
                            # bonanza!
                            arr[r][c] = 0
                            changed = True
    print('\n'.join(str_row(row) for row in arr))

    # Algorithm: if a 1 is surrounded by four 0's, keep it.
    # If a 1 is surrounded by three 0's, keep it.
    # If a 1 is surrounded by two 0's, at opposite edges, keep it.
    # If a 1 is next to only one 0,
    # If a 1 is entirely surrounded by 1's, keep it.

def attempt_level(lvl):
    level = board.Level(lvl)
    print(level.board)
    print(level.board.pairs_which_break())
    arr = [[1 if cell is board.Cell.EMPTY else 0 for cell in row] for row in level.board.cells]
    print(thin_array(arr))
    return False
    print("Obtained pairs which break.")

    breaking_pairs.sort(key=lambda tup1: abs(tup1[3]-tup1[4]))
    pairs = [x for x in breaking_pairs if x[3] <= x[4]]
    print("First pruning.")
    # Delete instances where filled is completely contained in another
    seen_arrs = []
    reduced_pairs = []
    for p in pairs:
        ignore = False
        for seen in seen_arrs:
            if seen.issubset(filled[(p[0], p[1], p[2])]):
                # can ignore filled
                ignore = True
                break
        if not ignore:
            seen_arrs.append(filled[(p[0], p[1], p[2])])
            reduced_pairs.append(p)

    print('\n'.join(str(c) for c in reduced_pairs))


    return False
    level.solve()
    return level.submit()

if __name__ == '__main__':
    for i in range(20,21):
        if not attempt_level(i):
            exit(1)
        print('{} done.'.format(i))
