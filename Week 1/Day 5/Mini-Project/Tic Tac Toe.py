'''
Mini-Project - Tic Tac Toe

Create a TicTacToe game in python, where two users can play together.

'''

# The uesr will choose the size of the board
board_size = 0

board = []

EMPTY = ' '
EX_SYMBOL = 'X'    # player 1
ZERO_SYMBOL = 'O'  # player 2

symbols = {1 : EX_SYMBOL, 2 : ZERO_SYMBOL}

we_have_a_winner = False


def display_board():

    print ("Tic Tac Toe")
    
    print("-"*((board_size-1)*8+7))
    
    for line in range(board_size-1):
        print ('   ', end="")
        print('   |   '.join(EMPTY for i in range(board_size)))
        print ('   ', end="")
        print('   |   '.join(board[line]))
        print ('   ', end="")
        print('   |   '.join(EMPTY for i in range(board_size)))
        print("-"*((board_size-1)*8+7))
    
    print ('   ', end="")
    print('   |   '.join(EMPTY for i in range(board_size)))
    print ('   ', end="")
    print('   |   '.join(board[board_size-1]))
    print ('   ', end="")
    print('   |   '.join(EMPTY for i in range(board_size)))
    
    print("-"*((board_size-1)*8+7))
    print("")
    


def player_input(player):
    
    while True:
        print(f"Player {player} turn: ({symbols[player]})")
        line = int(input("Enter a line: "))
        if line<1 or line>board_size:
            print("This line is out of limits.  Choose again.\n")
            continue
        column = int(input("Enter a column: "))
        if column<1 or column>board_size:
            print("This column is out of limits.  Choose again.\n")
            continue
        print("")
        if board[line-1][column-1] != EMPTY:
            print ("This cell is not empty.  Please choose an empty cell.\n")
        else:
            break

    board[line-1][column-1] = symbols[player]



def check_win():
    
    global we_have_a_winner

    # check horizontally
    for line in range(board_size):
        
        # check just the first cell in current line to see if it's empty or not
        if board[line][0] == EMPTY:
            continue  # continue to the next line

        all_match = True

        # continue from the second cell
        for column in range(1,board_size):
            if board[line][column] != board[line][column-1]:
                all_match = False
                break  # continue to the next line
        
        if all_match:
            we_have_a_winner = True
            return # completely break the current search - we have a winner


    # check vertically
    for column in range(board_size):
        
        # check just the first cell in current column to see if it's empty or not
        if board[0][column] == EMPTY:
            continue  # continue to the next column

        all_match = True

        # continue from the second cell
        for line in range(1,board_size):
            if board[line][column] != board[line-1][column]:
                all_match = False
                break  # continue to the next column
        
        if all_match:
            we_have_a_winner = True
            return # completely break the current search - we have a winner
        

    # check diagonally 1
    if board[0][0] != EMPTY:
        all_match = True
        for index in range(1, board_size):
            if board[index][index] != board[index-1][index-1]:
                all_match = False
                break
        if all_match:
            we_have_a_winner = True
            return # completely break the current search - we have a winner
        

    # check diagonally 2
    if board[board_size-1][0] != EMPTY:
        all_match = True
        for index in range(1, board_size):
            if board[(board_size-1)-index][index] != board[(board_size-1)-index+1][index-1]:
                all_match = False
                break
        if all_match:
            we_have_a_winner = True
            
        

def play():

    global we_have_a_winner
    global board_size

    print ("Welcome to Tic Tac Toe !\n")

    # The uesr will choose the size of the board
    while True:
        print("Please choose a board size.  It should be 2 or above.")
        board_size = int(input("What's the size of the board ? "))
        if board_size >= 2:
            print("")
            break
    
    # create an empty board
    for line in range(board_size):
        board.append([])
        for column in range(board_size):
            board[line].append(EMPTY)
            
    player = 1
    turns = 1  # count the number of turns in the game
    
    # start the game
    # continue the game as long as there's no winner and the board is not full
    while (not we_have_a_winner) and (turns <= board_size**2) :
        display_board()
        player_input(player)
        check_win()
        if we_have_a_winner:
            display_board()
            print(f"We have a winner - it's player number {player} ({symbols[player]}).\n")
            return
        turns += 1
        
        # switch player
        if player == 1:
            player = 2
        else:
            player = 1

    # check if there's a winner
    if not we_have_a_winner:
        print ("We have no winner.  It's a tie !\n")



play()

