'''

Create a deck of cards class.
The Deck of cards class should NOT inherit from a Card class.

The requirements are as follows:

The Card class should have a suit (Hearts, Diamonds, Clubs, Spades) and a value (A,2,3,4,5,6,7,8,9,10,J,Q,K)

The Deck class :
should have a shuffle method which makes sure the deck of cards has all 52 cards and then rearranges them randomly.
should have a method called deal which deals a single card from the deck. After a card is dealt, it should be removed from the deck.

'''

import random

class Card:
    
    def __init__ (self, suit, value):
        self.suit = suit
        self.value = value

    def __str__(self):
        return f"{self.suit} - {self.value}"


class Deck:

    def __init__ (self):

        self.suit_types = {'Hearts', 'Diamonds', 'Clubs', 'Spades'}
        self.value_types = {'A','2','3','4','5','6','7','8','9','10','J','Q','K'}
        
        self.cards = []  # a list of cards

        # initialize a deck package
        for suit in self.suit_types:
            for value in self.value_types:
                acard = Card(suit, value)  # create all possible cards
                self.cards.append(acard)  # add the card to the deck

    def shuffle(self):
        if len(self.cards) < 52:
            self.__init__()  # re-init the object - because the deck is not full - some cards were taken
        random.shuffle(self.cards)  # shuffle the cards
                

    def deal(self):
        if len(self.cards) > 0:  # if the deck is not empty
            acard = random.choice(self.cards)  # randomly choose a card from the list
            self.cards.remove(acard)  # remove it from the list
            return acard
        else:
            return None


d = Deck()
d.shuffle()
c = d.deal()
print (str(c))
c = d.deal()
print (str(c))
c = d.deal()
print (str(c))
d.shuffle()



