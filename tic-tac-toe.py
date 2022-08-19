import random
from math import inf as infinity


class Board:
    def __init__(self):
        self.turn_x = True
        self.setup_board()
        
    def setup_board(self):
        board = [['' for _ in range(3)] for _ in range(3)]
        self.board = board

    def get_winner(self):
        return self.winner

    def switch_turn(self):
        self.turn_x = not self.turn_x
    
    def random_start(self):
        row = random.randint(0, 2)
        column = random.randint(0,2)
        if self.turn_x:
            self.board[row][column] = 'x'
        else:
            self.board[row][column] = 'o'
        self.switch_turn()
        return True

    def check_who_win(self):
        for row in range(3):    
            if (self.board[row][0] == self.board[row][1] and self.board[row][1] == self.board[row][2]):       
                if (self.board[row][0] == 'x'):
                    return 10
                elif (self.board[row][0] == 'o'):
                    return -10
        for col in range(3) :
            if (self.board[0][col] == self.board[1][col] and self.board[1][col] == self.board[2][col]) :
                if (self.board[0][col] == 'x') :
                    return 10
                elif (self.board[0][col] == 'o') :
                    return -10
        if (self.board[0][0] == self.board[1][1] and self.board[1][1] == self.board[2][2]) :
            if (self.board[0][0] == 'x') :
                return 10
            elif (self.board[0][0] == 'o') :
                return -10

        if (self.board[0][2] == self.board[1][1] and self.board[1][1] == self.board[2][0]) :
            if (self.board[0][2] == 'x') :
                return 10
            elif (self.board[0][2] == 'o') :
                return -10
        return 0

    def update_winner(self):
        self.winner = self.check_who_win()
    
    def make_move(self, move: list):
        new_board = [[],[],[]]
        for i in range(3):
            for j in range(3):
                new_board[i].append(self.board[i][j])
        if self.turn_x:
            new_board[move[0]][move[1]] = 'x'
        else:
            new_board[move[0]][move[1]] = 'o'
        self.board = new_board
        self.update_winner()
        self.switch_turn()

    def delete_move(self, move: list):
        self.board[move[0]][move[1]] = ''
        self.update_winner()
        self.switch_turn()

    def heuristic(self):
        values_heuristic = [[3, 2, 3], [2, 4, 2], [3, 2, 3]]
        valx = 0
        valo = 0
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == "x":
                    valx += values_heuristic[i][j]
                if self.board[i][j] == "o":
                    valo += values_heuristic[i][j]
        if self.turn_x:
            valx = valo - valx
            return valx
        else:
            valo = valx - valo
            return valo

    def available_moves(self):
        moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == '':
                    moves.append([i, j])
        return moves

    def print_board(self):
        print(self.board[0][0] + " | " + self.board[0][1] + " | " + self.board[0][2])
        print("--+--+--")
        print(self.board[1][0] + " | " + self.board[1][1] + " | " + self.board[1][2])
        print("--+--+--")
        print(self.board[2][0] + " | " + self.board[2][1] + " | " + self.board[2][2] + "\n")

    def show_winner(self):
        if self.get_winner() == -10:
            print("O wins")
        if self.get_winner() == 10:
            print("X wins")
        else:
            print("Draw")
    
    def if_any_free(self):
        l = 0
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 'x' or self.board[i][j]:
                    l += 1
        if l == 9:
            return False
        return True
  
def minimax(board: Board, depth: int) -> int:
    if board.get_winner() != 0:
        return -board.get_winner() * 20
    if board.available_moves() == []:
        return 0
    if depth == 0:
        return board.heuristic()
    if board.turn_x:
        best_value = +infinity
    else:
        best_value = -infinity
    for move in board.available_moves():
        board.make_move(move)
        if board.turn_x:
            best_value = max(best_value, minimax(board, depth - 1))
        else:
            best_value= min(best_value, minimax(board, depth - 1))
        board.delete_move(move)
    return best_value

def get_best_move(board: Board, depth: int) -> list:
    if board.turn_x:
        best_value = +infinity
    else:
        best_value = -infinity
    best_move = [-10, -10]
    for move in board.available_moves():
        board.make_move(move)
        value = minimax(board, depth)
        if board.turn_x:
            if best_value < value:
                best_value = value
                best_move = move
        else:
            if best_value > value:
                best_value = value
                best_move = move
        board.delete_move(move)
    return best_move

def tictactoe(depth: int):
    board = Board()
    board.print_board()
    board.random_start()
    board.print_board()
    while(board.check_who_win() == 0 and board.if_any_free() == True):
        if board.check_who_win() == 0:
            move = get_best_move(board, depth)
            board.make_move(move)
            board.print_board()
    board.show_winner()

if __name__ == "__main__":
    tictactoe(depth = 5)
