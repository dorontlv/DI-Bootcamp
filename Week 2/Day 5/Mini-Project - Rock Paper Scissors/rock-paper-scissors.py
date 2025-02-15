import game


def get_user_menu_choice():
    print("")
    print ("Menu:")
    print ("(p) Play a new game")
    print ("(s) Show scores")
    print ("(q) Quit")
    return input("What do you want to do ? ").lower()



def print_results(results:dict):  # {'won':x , lost:y , drew:z}
    for k,v in results.items():
        print(f"You {k} : {v} times")
    print("")



def main():
    
    g = game.Game()

    results = {'won':0 , 'lost':0 , 'drew':0}

    while True:

        while True:        
            selection = get_user_menu_choice()
            if selection and (selection in "psq"):
                break
            
        if selection == 'q':
            print("")
            print ("Bye Bye ...")
            return
        
        if selection == 's':
            print("")
            print_results(results)
        else:
            result = g.play()

            results[result] += 1

        
main()

