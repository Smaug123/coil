#!/usr/bin/env python3

import requests

import board


def attempt_level(lvl):
    r = requests.post('http://www.hacker.org/coil/index.php',
                      {'name': 'laz0r',
                       'spw': '03233c19b6de691fd1806eb1aff59f6a',
                       'go': 'Go To Level',
                       'gotolevel': lvl})
    print("Received.")
    html_str = r.text
    this_board = board.Board(html_str)

    this_board.solve()
    #cProfile.runctx('this_board.solve()', {'this_board': this_board}, locals={})

    if this_board.is_solved():
        print("Sending.")
        r = requests.post('http://www.hacker.org/coil/index.php',
                          {'name': 'laz0r',
                           'spw': '03233c19b6de691fd1806eb1aff59f6a',
                           'path': this_board.solution(),
                           'y': this_board.start_row,
                           'x': this_board.start_col
                           })
        assert(r.text != "<br>your solution sucked<br><a href=index.php>back to puzzle</a>")
        return True
    return False

if __name__ == '__main__':
    for i in range(25,60):
        if not attempt_level(i):
            exit(1)
        print('{} done.'.format(i))
