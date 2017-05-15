#!/usr/bin/env python3

import board


def attempt_level(lvl):
    level = board.Level(lvl)
    level.solve()
    return level.submit()

if __name__ == '__main__':
    for i in range(5,60):
        if not attempt_level(i):
            exit(1)
        print('{} done.'.format(i))
