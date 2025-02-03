'''
Daily Challenge : Lists & Strings

'''

print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Challenge 1")


number = int(input("Enter a number: "))
lengh = int(input("Enter a lengh: "))
result = 0

for item in range(lengh):
    result += number
    print (result, end=" ")

print ("")


print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print ("Challenge 2")


string = input("Please enter a sentence: ")

last_printed_chr = []

# always compare the curreent char to the last character we printed - if it's different then print it
for current_chr in string:
    if current_chr != last_printed_chr:
        print(current_chr, end="")
        last_printed_chr = current_chr

print("")


