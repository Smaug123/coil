import re
import copy
import enum

import requests


# Pure Python, usable speed but over 10x greater runtime than Cython version
def fill(data, start_coords, fill_value):
    """
    Flood fill algorithm

    Parameters
    ----------
    data : (M, N) ndarray of uint8 type
        Image with flood to be filled. Modified inplace.
    start_coords : tuple
        Length-2 tuple of ints defining (row, col) start coordinates.
    fill_value : int
        Value the flooded area will take after the fill.

    Returns
    -------
    None, ``data`` is modified inplace.
    """
    xsize, ysize = len(data), len(data[0])
    orig_value = data[start_coords[0]][start_coords[1]]

    stack = {(start_coords[0], start_coords[1])}
    if fill_value == orig_value:
        raise ValueError("Filling region with same value "
                         "already present is unsupported. "
                         "Did you already fill this region?")

    while stack:
        x, y = stack.pop()

        if data[x][y] == orig_value:
            data[x][y] = fill_value
            if x > 0:
                stack.add((x - 1, y))
            if x < (xsize - 1):
                stack.add((x + 1, y))
            if y > 0:
                stack.add((x, y - 1))
            if y < (ysize - 1):
                stack.add((x, y + 1))


@enum.unique
class Cell(enum.Enum):
    """
    A cell in a Mortal Coil board.
    """
    EMPTY = '.'
    VISITED = 'O'
    CURRENT = '@'
    BLOCKED = 'X'

    def __str__(self):
        return self.value

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


@enum.unique
class Direction(enum.Enum):
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4

    def __str__(self):
        return self.name[0]

class DirectionError(Exception):
    pass


class Board:
    """
    A Mortal Coil board: http://hacker.org/coil/

    Attributes:
      height: the number of rows in the board
      width: the number of columns in the board
      cells: grid of Cell objects, as a list of rows
      row, col: current row number and column number, zero-indexed
      moves: list of moves
    """

    def _new_indices(self, row, col, direction):
        """
        Return the new coordinates if we go in a given direction.

        Result is a pair (row, col)
        """
        if direction is Direction.LEFT:
            if col == 0:
                return False
            return (row, col-1)
        elif direction is Direction.RIGHT:
            if col == self.width-1:
                return False
            return (row, col+1)
        elif direction is Direction.DOWN:
            if row == self.height-1:
                return False
            return (row+1, col)
        elif direction is Direction.UP:
            if row == 0:
                return False
            return (row-1, col)
        else:
            assert(False)

    def state(self, row=None, col=None):
        if row is None:
            row = self.row
        if col is None:
            col = self.col
        return self.cells[row][col]

    def set_state(self, row, col, val):
        self.cells[row][col] = val

    def __init__(self, html):
        """
        New board.

        :param html: HTML output of the server request
        """
        self.row = None
        self.col = None

        groups = re.search(r'FlashVars" value="x=(.+)&y=(.+)&board=(.+)"', html)
        self.height = int(groups.group(2))
        self.width = int(groups.group(1))

        board_str = groups.group(3)
        self.cells = [[Cell(c) for c in row] for row in chunks(board_str, self.width)]
        self.start_board = copy.deepcopy(self.cells)

        self._allowed_directions = None
        self.set_up_directions()
        self.moves = []

    def __str__(self):
        return '\n'.join(''.join(str(c) for c in row) for row in self.cells)

    def set_up_directions(self):
        """
        Initialise self._allowed_directions by setting its value for every cell.

        self._allowed_directions is the internal cache of which directions are
        available from each square. This function initialises the cache.
        Only adds to the existing directions; never removes. This is for
        speed.
        """
        if self._allowed_directions is None:
            self._allowed_directions = [[set() for col in range(self.width)] for row in range(self.height)]
        for row in range(self.height):
            for col in range(self.width):
                for direction in Direction:
                    if direction not in self._allowed_directions[row][col]:
                        res = self._new_indices(row, col, direction)
                        if res:
                            newrow, newcol = res
                            if self.state(newrow, newcol) is Cell.EMPTY:
                                self._allowed_directions[row][col].add(direction)

    def allowed_directions(self, rownum=None, colnum=None):
        """
        Get the directions we may travel from position (rownum, colnum).

        Both these arguments default to "the current position".
        Returns a set of the Direction objects.
        """
        if rownum is None:
            rownum = self.row
        if colnum is None:
            colnum = self.col

        return self._allowed_directions[rownum][colnum]

    def undo(self, depth):
        """
        Undo, so that we've only made <depth> moves
        """
        moves = self.moves[:depth]
        self.set_start(self.start_row, self.start_col)
        for move in moves:
            self.move_direction(move)

    def clear_current(self):
        """
        Resets the board to empty state.
        """
        self.cells = copy.deepcopy(self.start_board)

    def set_start(self, row, col):
        """
        Empty the board and start at the given position.

        Clears the current moves; sets the start_row and start_col,
        as well as the row and col; empties the board; resets
        the cache of directions; sets the current row, col state to CURRENT.
        """
        self.moves = []
        self.start_row = row
        self.start_col = col
        self.row = row
        self.col = col
        self.clear_current()
        self.set_up_directions()
        self.set_state(row, col, Cell.CURRENT)

    def move_direction(self, direction):
        """
        Move in the given direction from the current position.
        """
        assert(direction in self.allowed_directions(self.row, self.col))
        # Repeatedly move in that direction, setting the allowed_directions of the cells we go through
        self.moves.append(direction)
        while True:
            res = self._new_indices(self.row, self.col, direction)
            if not res:
                break
            new_row, new_col = res
            if self.state(new_row, new_col) in (Cell.BLOCKED, Cell.VISITED):
                break

            self.set_state(self.row, self.col, Cell.VISITED)
            if self.col < self.width-1 and Direction.LEFT in self._allowed_directions[self.row][self.col+1]:
                self._allowed_directions[self.row][self.col+1].remove(Direction.LEFT)
            if self.col > 0 and Direction.RIGHT in self._allowed_directions[self.row][self.col-1]:
                self._allowed_directions[self.row][self.col-1].remove(Direction.RIGHT)
            if self.row < self.height - 1 and Direction.UP in self._allowed_directions[self.row+1][self.col]:
                self._allowed_directions[self.row+1][self.col].remove(Direction.UP)
            if self.row > 0 and Direction.DOWN in self._allowed_directions[self.row-1][self.col]:
                self._allowed_directions[self.row-1][self.col].remove(Direction.DOWN)
            self.row, self.col = new_row, new_col
            self.set_state(self.row, self.col, Cell.CURRENT)

    def make_trivial_forced_moves(self):
        """
        Make moves that are instantly forced from the current position, until we can't
        """
        assert(self.state() is Cell.CURRENT)
        made_a_change = False

        while True:
            directions = self.allowed_directions(self.row, self.col)
            if len(directions) == 1:
                (direction,) = directions
                self.move_direction(direction)
                made_a_change = True
            else:
                return made_a_change

    def is_solved(self):
        for slice in self.cells:
            for c in slice:
                if c not in (Cell.VISITED, Cell.BLOCKED, Cell.CURRENT):
                    return False
        return True

    def is_disconnected(self):
        # Perform flood fill
        binary = [[1 for r in range(self.width)] for s in range(self.height)]
        start_i = None
        start_j = None
        for i in range(self.height):
            for j in range(self.width):
                if self.state(i, j) is Cell.EMPTY:
                    if start_i is None:
                        start_i = i
                        start_j = j
                    binary[i][j] = 0

        if start_i is None:
            # the grid is actually filled
            return False
        fill(binary, (start_i, start_j), 2)

        for i in range(len(binary)):
            for j in range(len(binary[i])):
                if binary[i][j] == 0:
                    return True

        return False

    def contains_two_deadends(self):
        deadend = None
        for row in range(self.height):
            for col in range(self.width):
                if self.state(row, col) is Cell.EMPTY:
                    if len(self.allowed_directions(row, col)) <= 1:
                        if deadend:
                            return (deadend, (row, col))
                        else:
                            deadend = (row, col)
        return False

    def make_deadend_forced_moves(self):
        # If there are two dead ends, and we can move immediately into one of them, we must do so
        de = self.contains_two_deadends()
        if de:
            # We can't move in a non-deadend direction
            for direction in list(self.allowed_directions(self.row, self.col)):
                newrow, newcol = self._new_indices(self.row, self.col, direction)
                if (newrow, newcol) not in de:
                    self._allowed_directions[self.row][self.col].remove(direction)
            return self.make_trivial_forced_moves()
        else:
            return False

    def make_forced_moves(self):
        changed = True
        while changed:
            changed = self.make_trivial_forced_moves() or self.make_deadend_forced_moves()

    def cannot_move(self):
        return len(self.allowed_directions()) == 0

    def unsolvable(self):
        # To be called after we have finished making all the moves which are forced
        return self.cannot_move() or self.is_disconnected()

    def attempt_solution(self):
        """
        In a position where we could make one of several possible actions:
          if we can prove all actions lead to failure, reset the grid to current state and return False
          if we succeed, return True
        """
        curr_stack_lvl = len(self.moves)
        for direction in list(self.allowed_directions()):
            self.move_direction(direction)
            self.make_forced_moves()
            if not self.is_solved() and not self.unsolvable():
                if self.attempt_solution():
                    return True
            elif self.is_solved():
                return True

            self.undo(depth=curr_stack_lvl)

        return False

    def solution(self):
        return ''.join(str(d) for d in self.moves)


class Level:

    def __init__(self, level):
        r = requests.post('http://www.hacker.org/coil/index.php',
                          {'name': 'laz0r',
                           'spw': '03233c19b6de691fd1806eb1aff59f6a',
                           'go': 'Go To Level',
                           'gotolevel': level})
        html_str = r.text
        self.board = Board(html_str)

    def submit(self):
        if self.board.is_solved():
            r = requests.post('http://www.hacker.org/coil/index.php',
                              {'name': 'laz0r',
                               'spw': '03233c19b6de691fd1806eb1aff59f6a',
                               'path': self.board.solution(),
                               'y': self.board.start_row,
                               'x': self.board.start_col
                               })
            assert(r.text != "<br>your solution sucked<br><a href=index.php>back to puzzle</a>")
            return True
        return False

    def solve(self):
        for row in range(self.board.height):
            for col in range(self.board.width):
                if self.board.state(row, col) is not Cell.EMPTY:
                    continue

                self.board.set_start(row, col)
                if self.board.attempt_solution():
                    return True


def board_to_str(brd):
    return '\n'.join((''.join(str(c) for c in row) for row in brd))
