import anagram_checker


ac = anagram_checker.AnagramChecker()

while True:

    # printing a menu
    while True:        
            print ("Menu:")
            print ("(W) choose a word")
            print ("(Q) Quit")
            selection = input("What do you want to do ? ").lower()
            if selection and (selection in "wq"):
                break

    if selection == 'q':
        break  # end the program

    # getting the word, and validating it
    while True:
        print("")
        word = input("enter a word: ").lower()
        print("")
        word_chars = list(word)
        
        # check that is only alphanumeric
        continue_while = False
        for chr in word_chars:
             if (chr!=' ') and not chr.isalpha():
                continue_while = True
                print ("Only alphanumeric values are allowed.")
                break
        if continue_while:
            continue

        while word_chars[0] == ' ':  # remove spaces at the begining of the word
            word_chars.pop(0)

        while word_chars[len(word_chars)-1] == ' ':  # remove spaces at the end of the word
            word_chars.pop(len(word_chars)-1)

        word = ''.join(word_chars)
        
        if ' ' in word:
            print ("Please anter only one word.")
        else:
            break

    
    # at this point - the word is validated
    
    anagrams = ac.get_anagrams(word)  # get all possible anagrams, even if it's not a valid english word

    valid_anagrams = [anagram for anagram in anagrams if ac.is_valid_word(anagram)]  # get only english words from the anagrams list
    
    if not valid_anagrams:  # the entire list is empty
        print ("The word you entered, including all its anagrams, is not english.\n")
        continue

    # remove our own word from the list of anagrams
    if word in valid_anagrams:
        valid_anagrams.remove(word)

    print (f"Your word is: {word}")
    if (valid_anagrams):
        print ("The anagrams for your word are:")
        print(', '.join(valid_anagrams))
    else:
         print("There are no anagrams for your word.")
         
    print ("")