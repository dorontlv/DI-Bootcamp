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

encrypted_string_with_newline = '''7ii
Tsx
h%?
i #
sM 
$a 
#t%
^r!
'''

encrypted_string = ""

# just to remove the newline in the string (although maybe there is a function in python that can do this)
for chr in encrypted_string_with_newline:
    if chr != "\n":
        encrypted_string += chr

print("This is the matrix string:")
print (encrypted_string)

Matrix = []
index = 0

# put the string into a matrix
for line in range(8):
    Matrix.append([])
    for column in range(3):
        Matrix[line].append(encrypted_string[index])
        index += 1

print("This is the matrix:")
print (Matrix)

matrix_string_decrypted_message = ""
letter = False

# decrypt the message
for line in range(len(Matrix[0])):  # 3
    for column in range(len(Matrix)):  # 8
        chr = Matrix[column][line]
        if chr>='a' and chr<='z' or chr>='A' and chr<='Z':
            matrix_string_decrypted_message += chr
            letter = True
        elif letter:
            matrix_string_decrypted_message += ' '
            letter = False
 
print(f"The matrix decrypted message is:\n{matrix_string_decrypted_message}")

