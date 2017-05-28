import os
import re
import copy
import enum
import errno
import itertools

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
    
    VISITED_LEFT, for instance, means we left this square leftwards.
    """
    EMPTY = '.'
    VISITED_START = 'O'
    VISITED_LEFT = '<'
    VISITED_RIGHT = '>'
    VISITED_UP = '^'
    VISITED_DOWN = 'v'
    CURRENT = '@'
    BLOCKED = 'X'

    def __str__(self):
        return self.value

def can_visit(state):
    return state is Cell.EMPTY

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


def reverse_dir(direction):
    if direction is Direction.LEFT:
        return Direction.RIGHT
    elif direction is Direction.RIGHT:
        return Direction.LEFT
    elif direction is Direction.UP:
        return Direction.DOWN
    elif direction is Direction.DOWN:
        return Direction.UP
    else:
        assert(False)


def perpendicular(dir1, dir2):
    if dir1 is Direction.LEFT or dir1 is Direction.RIGHT:
        return dir2 is Direction.UP or dir2 is Direction.DOWN
    if dir1 is Direction.UP or dir1 is Direction.DOWN:
        return dir2 is Direction.LEFT or dir2 is Direction.RIGHT


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

        Ignores whether or not a square is blocked.
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
        for direc in Direction:
            new_indices = self._new_indices(row, col, direc)
            if new_indices:
                if val == Cell.EMPTY:
                    self._allowed_directions[new_indices[0]][new_indices[1]].add(reverse_dir(direc))
                else:
                    self._allowed_directions[new_indices[0]][new_indices[1]] -= {reverse_dir(direc)}

    @classmethod
    def from_html(cls, html):
        """
        Creates a board from the HTML input.
        
        Returns the board.
        """
        groups = re.search(r'FlashVars" value="x=(.+)&y=(.+)&board=(.+)"', html)
        board_str = groups.group(3)
        height = int(groups.group(2))
        width = int(groups.group(1))
        return Board(height=height,
                     width=width,
                     cells=[[Cell(c) for c in row] for row in chunks(board_str, width)])

    def __init__(self, height, width, cells):
        """
        New board.
        """
        self.row = None
        self.col = None
        self.height = height
        self.width = width
        self.cells = cells

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
                            if can_visit(self.state(newrow, newcol)):
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

    def undo_single_move(self):
        """
        Undoes a move, changing allowed_moves, moves, and the board.
        """
        prev_move = self.moves[-1]
        back_dir = reverse_dir(prev_move)
        self.moves = self.moves[:-1]
        # from current position, walk in the opposite direction to prev_move
        # until we reach a square whose state is not prev_move
        cellstate = Cell['VISITED_{}'.format(prev_move.name)]
        self.set_state(self.row, self.col, cellstate)
        while self.state().name == 'VISITED_{}'.format(prev_move.name):
            new_ind = self._new_indices(self.row, self.col, back_dir)
            if new_ind and self.state(*new_ind) == cellstate:
                self.set_state(self.row, self.col, Cell.EMPTY)
                self.row, self.col = new_ind
            else:
                break

        self.set_state(self.row, self.col, Cell.CURRENT)

    def undo(self, depth):
        """
        Undo, so that we've only made <depth> moves
        """
        for i in range(len(self.moves)-depth):
            self.undo_single_move()

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
            if not can_visit(self.state(new_row, new_col)):
                break

            if direction is Direction.RIGHT:
                self.set_state(self.row, self.col, Cell.VISITED_RIGHT)
            elif direction is Direction.LEFT:
                self.set_state(self.row, self.col, Cell.VISITED_LEFT)
            elif direction is Direction.UP:
                self.set_state(self.row, self.col, Cell.VISITED_UP)
            elif direction is Direction.DOWN:
                self.set_state(self.row, self.col, Cell.VISITED_DOWN)
            else:
                assert(False)

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
                if can_visit(c):
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


    def pairs_which_break(self):
        """
        Returns a list of pairs of squares which mutually disconnect the board.
        
        The result is a list of ((row1, col1), (row2, col2), direction, size of one chunk, size of other chunk).
        """
        cel = self.cells_with_two()
        # cel is the cells with two neighbours.
        # It's a broken assumption that only these can disconnect the grid.
        # For example, there can be a bottleneck of two cells next to
        # each other, whose removal together disconnects the graph.

        # For now, keep the cells which are a bend and which don't have an empty
        # cell in the elbow of the bend.
        pruned = []
        for c in cel:
            dirs = list(self.allowed_directions(*c))
            if perpendicular(*dirs):
                move1 = self._new_indices(c[0], c[1], dirs[0])
                move2 = self._new_indices(move1[0], move1[1], dirs[1])
                if self.state(*move2) is Cell.EMPTY:
                    # can't disconnect with this one
                    continue
                pruned.append(c)

        # For each pair of cells, do they disconnect?
        breaking_pairs = []
        pairs = itertools.combinations(pruned, 2)
        filled = {}
        num_squares = len({(i, j) for i, row in enumerate(self.cells) for j, c in enumerate(row) if c is Cell.EMPTY})
        for c1, c2 in pairs:
            dirs = list(self.allowed_directions(*c1))
            for d in dirs:
                co = copy.deepcopy(self.cells)
                co[c1[0]][c1[1]] = Cell.BLOCKED
                co[c2[0]][c2[1]] = Cell.BLOCKED
                new = self._new_indices(c1[0], c1[1], d)
                if new == c2:
                    break
                fill(co, new, Cell.CURRENT)
                filled[(c1, c2, d)] = {(i, j) for i, row in enumerate(co) for j, c in enumerate(row) if c is Cell.CURRENT}
                num_current = len(filled[(c1, c2, d)])
                num_empty = num_squares - num_current - 2  # 2 already blocked

                if num_empty >= 1:
                    breaking_pairs.append((c1,c2, d, num_empty, num_current))

        # Sort them. Also, each pair has appeared twice, once for each
        # direction out of the first cell; remove the duplicates.
        breaking_pairs.sort(key=lambda tup1: abs(tup1[3]-tup1[4]))
        pairs = [x for x in breaking_pairs if x[3] <= x[4]]

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
        return reduced_pairs

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

    def cells_with_two(self):
        """
        Returns a list of (row, col) where the cells in those positions
        have exactly two directions out of which to go.
        """
        ans = []
        for row in range(self.height):
            for col in range(self.width):
                if len(self.allowed_directions(row, col)) == 2:
                    ans.append((row, col))
        return ans


class Level:
    _CONFIG_DIR = os.path.join(os.path.expanduser('~'), '.mortalcoil')
    _CACHE_DIR = os.path.join(_CONFIG_DIR, 'cache')
    _SPW_FILE = os.path.join(_CONFIG_DIR, 'spw.txt')

    @staticmethod
    def get_cache_name(level):
        """
        Gets the file name corresponding to the given level.
        """
        return os.path.join(Level._CACHE_DIR, str(level) + '.html')

    @staticmethod
    def get_spw():
        if not os.path.exists(Level._SPW_FILE):
            spw = input("Enter your SPW from http://www.hacker.org/util/getsubmitpw.php:\n")
            with open(Level._SPW_FILE, 'w') as f:
                f.write(spw)
            return spw
        else:
            with open(Level._SPW_FILE) as f:
                return f.read().decode("ascii")

    @staticmethod
    def cache_level(level):
        """
        Stores a level.
        
        level should be the integer number of the level.
        """
        cache_file = Level.get_cache_name(level)

        # Make the directories containing the cache file
        dirpath = os.path.dirname(cache_file)
        try:
            os.makedirs(dirpath)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(dirpath):
                pass
            else:
                raise

        r = requests.post('http://www.hacker.org/coil/index.php',
                          {'name': 'laz0r',
                           'spw': Level.get_spw(),
                           'go': 'Go To Level',
                           'gotolevel': level},
                          stream=True)
        with open(cache_file, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

    @staticmethod
    def get_cached_level(level):
        with open(Level.get_cache_name(level)) as f:
            return f.read()

    @staticmethod
    def level_is_known(level):
        return os.path.exists(Level.get_cache_name(level))

    def __init__(self, level):
        if not self.level_is_known(level):
            self.cache_level(level)
        html_str = self.get_cached_level(level)
        self.board = Board.from_html(html_str)

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
        print(self.board)
        for row in range(self.board.height):
            for col in range(self.board.width):
                if self.board.state(row, col) is not Cell.EMPTY:
                    continue

                self.board.set_start(row, col)
                if self.board.attempt_solution():
                    print(self.board)
                    print(self.board.solution())
                    print(self.board.start_row)
                    print(self.board.start_col)
                    return True


def board_to_str(brd):
    return '\n'.join((''.join(str(c) for c in row) for row in brd))
