import random

class Game:

    def __init__(self):
        self.rps_options = ['rock', 'paper', 'scissors']


    def get_user_item(self):
        # Ask the user to select an item (rock/paper/scissors). Keep asking until the user has selected one of the items – use data validation and looping. Return the item at the end of the function.
        while True:
            print("")
            selection = input("choose (R)ock, (P)aper, (S)cissors: ").lower()
            if selection and (selection in "rps"):
                break
        
        if selection == 'r':
            return 'rock'
        if selection == 'p':
            return 'paper'
        if selection == 's':
            return 'scissors'   


    def get_computer_item(self):
        # Select rock/paper/scissors at random for the computer. Return the item at the end of the function.
        return random.choice(self.rps_options)


    def get_game_result(self, user_item, computer_item):
        # Determine the result of the game.
        # user_item – the user’s chosen item (rock/paper/scissors)
        # computer_item – the computer’s chosen (random) item (rock/paper/scissors)
        # Return either win, draw, or loss
        
        if user_item == 'rock':
            if computer_item == 'paper':
                return 'lost'
            elif computer_item == 'scissors':
                return 'won'
            else:
                return 'drew'
            
        if user_item == 'paper':
            if computer_item == 'rock':
                return 'won'
            elif computer_item == 'scissors':
                return 'lost'
            else:
                return 'drew'
            
        if user_item == 'scissors':
            if computer_item == 'paper':
                return 'won'
            elif computer_item == 'rock':
                return 'lost'
            else:
                return 'drew'
            
        

    def play(self):
        user_choice = self.get_user_item()
        computer_choice = self.get_computer_item()
        result = self.get_game_result(user_choice, computer_choice)
        
        print("")
        print(f"You selected {user_choice}.  The computer selected {computer_choice}.  You {result}.")
        
        return result



