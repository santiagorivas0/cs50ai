import itertools
import random


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def known_mines(self):
        if len(self.cells) == self.count:
            return set(self.cells)
        return set()

    def known_safes(self):
        if self.count == 0:
            return set(self.cells)
        return set()

    def mark_mine(self, cell):
        if cell in self.cells:
            self.cells.remove(cell)
            self.count -= 1

    def mark_safe(self, cell):
        if cell in self.cells:
            self.cells.remove(cell)



class MinesweeperAI():
    def __init__(self, height=8, width=8):
        self.height = height
        self.width = width
        self.moves_made = set()
        self.mines = set()
        self.safes = set()
        self.knowledge = []

    def mark_mine(self, cell):
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def add_knowledge(self, cell, count):
        self.moves_made.add(cell)
        self.mark_safe(cell)

        neighbors = set()
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):
                if (0 <= i < self.height and 0 <= j < self.width and (i, j) != cell):
                    if (i, j) in self.mines:
                        count -= 1
                    elif (i, j) not in self.safes:
                        neighbors.add((i, j))

        self.knowledge.append(Sentence(neighbors, count))
        self.update_knowledge()

    def update_knowledge(self):
        updated = True
        while updated:
            updated = False
            safes = set()
            mines = set()

            for sentence in self.knowledge:
                safes |= sentence.known_safes()
                mines |= sentence.known_mines()

            for safe in safes:
                if safe not in self.safes:
                    self.mark_safe(safe)
                    updated = True

            for mine in mines:
                if mine not in self.mines:
                    self.mark_mine(mine)
                    updated = True

            new_sentences = []
            for s1 in self.knowledge:
                for s2 in self.knowledge:
                    if s1 != s2 and s1.cells.issubset(s2.cells):
                        inferred = Sentence(s2.cells - s1.cells, s2.count - s1.count)
                        if inferred not in self.knowledge and inferred not in new_sentences:
                            new_sentences.append(inferred)

            self.knowledge.extend(new_sentences)
            self.knowledge = [s for s in self.knowledge if len(s.cells) > 0]

    def make_safe_move(self):
        for move in self.safes:
            if move not in self.moves_made:
                return move
        return None

    def make_random_move(self):
        choices = [
            (i, j) for i in range(self.height) for j in range(self.width)
            if (i, j) not in self.moves_made and (i, j) not in self.mines
        ]
        return random.choice(choices) if choices else None
