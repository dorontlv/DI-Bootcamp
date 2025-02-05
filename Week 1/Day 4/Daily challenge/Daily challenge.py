'''

Daily challenge: Solve the Matrix

Given a “Matrix” string:

7ii
Tsx
h%?
i #
sM 
$a 
#t%
^r!

To decrypt the matrix, Neo reads each column from top to bottom, starting from the leftmost column, selecting only the alpha characters and connecting them.
Then he replaces every group of symbols between two alpha characters by a space.

Using his technique, try to decode this matrix.


'''

Matrix = [
['7','i','i'],
['T','s','x'],
['h','%','?'],
['i',' ','#'],
['s','M',' '],
['$','a',' '],
['#','t','%'],
['^','r','!']
]

matrix_string = ""
matrix_string_decrypted_message = ""
letter = False

# go over the matrix - read the matrix message
for line in range(len(Matrix[0])):
    for column in range(len(Matrix)):
        matrix_string += Matrix[column][line]

print(f"The matrix message is: {matrix_string}")

# decrypt the message
for chr in matrix_string:
    if chr>='a' and chr<='z' or chr>='A' and chr<='Z':
        matrix_string_decrypted_message += chr
        letter = True
    elif letter:
            matrix_string_decrypted_message += ' '
            letter = False

print(f"The matrix decrypted message is: {matrix_string_decrypted_message}")


