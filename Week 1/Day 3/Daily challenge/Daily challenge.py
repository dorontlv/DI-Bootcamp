'''
Daily Challenge: Dictionaries

'''

print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Challenge 1")


'''

Challenge 1

"dodo" ➞ { "d": [0, 2], "o": [1, 3] }

"froggy" ➞ { "f":  [0], "r": [1], "o": [2], "g": [3, 4], "y": [5] }

"grapes" ➞ { "g": [0], "r": [1], "a": [2], "p": [3]}

'''

word = input ("Please enter a word: ")

letters = {}
index = 0

for letter in word:
    if letter not in letters:
        # when it's a new letter - add it into a new list (with its index) and add the list into the dict
        a_list = []
        a_list.append(index)
        letters[letter] = a_list
    else:
        # a letter that is already in the dictionary - add its index to the list
        letters[letter].append(index)
    index += 1

print (letters)



print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Challenge 2")



'''
Challenge 2

1. Create a program that prints a list of the items you can afford in the store with the money you have in your wallet.
2. Sort the list in alphabetical order.
3. Return “Nothing” if you can’t afford anything from the store.

items_purchase = {
  "Water": "$1",
  "Bread": "$3",
  "TV": "$1,000",
  "Fertilizer": "$20"
}

wallet = "$300"

➞ ["Bread", "Fertilizer", "Water"]

items_purchase = {
  "Apple": "$4",
  "Honey": "$3",
  "Fan": "$14",
  "Bananas": "$4",
  "Pan": "$100",
  "Spoon": "$2"
}

wallet = "$100" 

➞ ["Apple", "Bananas", "Fan", "Honey", "Spoon"]

# In fact the prices of Apple + Honey + Fan + Bananas + Pan is more that $100, so you cannot by the Pan, 
# instead you can by the Spoon that is $2

items_purchase = {
  "Phone": "$999",
  "Speakers": "$300",
  "Laptop": "$5,000",
  "PC": "$1200"
}

wallet = "$1" 

➞ "Nothing"

'''


items_purchase = {
  "Water": "$1",
  "Bread": "$3",
  "TV": "$1,000",
  "Fertilizer": "$20"
}

wallet = "$300"

# creating a sorted list of the dict keys
sorted_keys = sorted(items_purchase)

# a dict that will contain the original dict, but it will be ordered (because we are iterating through an ordered list)
sorted_items_purchase = {}

# removing the , and $ signs from the price, and converting it into an integer.
# and then put it into the new dict (so we will get a sorted dict).
for food in sorted_keys:
    cost = ""
    for chr in items_purchase[food]:
        if chr not in "$,":
            cost += chr
    sorted_items_purchase[food] = int(cost)

# removing the , and $ signs from the wallet
wallet_with_numbers_only = ""
for chr in wallet:
    if chr not in "$,":
            wallet_with_numbers_only += chr

# converting it into an int
money_I_have = int(wallet_with_numbers_only)

# calculate which products we can afford to buy
spent_money = 0
products_I_buy = []
for item, cost in sorted_items_purchase.items():
     if (spent_money + cost <= money_I_have):
          spent_money += cost
          products_I_buy.append(item)

# check to see if we bought something or not
if spent_money > 0:
    print (products_I_buy)
else:
     print ("Nothing")


