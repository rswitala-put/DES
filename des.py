import tkinter as tk
import base64
import pyaudio
import wave

# ''' generator '''------------------------------------------------------------------------------------

r = 31
m = 397
n = 624
w = 32
a = 0x9908B0DF
u = 11
d = 0xFFFFFFFF
s = 7
b = 0x9D2C5680
t = 15
c = 0xEFC60000
l = 18
f = 1812433253

my_number = [0 for i in range(n)]
index = n + 1
lower_mask = 0x7FFFFFFF
upper_mask = 0x80000000


def seed(seed):
    my_number[0] = seed
    for i in range(1, n):
        temp = f * (my_number[i - 1] ^ (my_number[i - 1] >> (w - 2))) + i
        my_number[i] = temp & 0xff  # zmiana z 8 bitów na 32 w zależności od długości maski


def get_number():
    global index
    if index >= n:
        mix()
        index = 0

    y = my_number[index]
    y = y ^ (y >> u)
    y = y ^ ((y << t) & c)
    y = y ^ ((y << s) & b)
    y = y ^ (y >> l)

    index += 1
    return y & 0xff  # zmiana z 8 bitów na 32 w zależności od długości maski


def mix():
    for i in range(0, n):
        x = (my_number[i] & upper_mask) + (my_number[(i + 1) % n] & lower_mask)
        x_bit = x >> 1
        if (x % 2) != 0:
            x_bit = x_bit ^ a
        my_number[i] = my_number[(i + m) % n] ^ x_bit


def generate_key():
    print("Uruchamiam generator liczb")
    CHUNK = 1024
    FORMAT = pyaudio.paInt16  # rozmiar probki w bajtach, przy innych  duzo zer prawdopodobnie kodowanie wav -> chunkow
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "output_2.wav"
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("*** recording ***")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'rb')
    binary_data = wf.readframes(wf.getnframes())

    a = int.from_bytes(binary_data[15000:15025], byteorder="big")
    mask = 255
    seed_ = int(a) & mask
    print("seed= " + str(seed_))
    wf.close()
    seed(seed_)
    number = []

    for i in range(8):
        tmp = get_number()
        tmp = chr(tmp)
        number.append(tmp)
    return number


# ----------------------------------------------------MATRIX/VECTORS-----------------------------------------------------

# Initial permut matrix for the datas
IP = [58, 50, 42, 34, 26, 18, 10, 2,
      60, 52, 44, 36, 28, 20, 12, 4,
      62, 54, 46, 38, 30, 22, 14, 6,
      64, 56, 48, 40, 32, 24, 16, 8,
      57, 49, 41, 33, 25, 17, 9, 1,
      59, 51, 43, 35, 27, 19, 11, 3,
      61, 53, 45, 37, 29, 21, 13, 5,
      63, 55, 47, 39, 31, 23, 15, 7]

# This table specifies the input permutation on a 64-bit block. The meaning is as follows: the first bit of the output
# is taken from the 58th bit of the input; the second bit from the 50th bit, and so on, with the last bit of the output
# taken from the 7th bit of the input. This information is presented as a table for ease of presentation; it is a
# vector, not a matrix.

# Final permut for datas after the 16 rounds IP^-1
IP_1 = [40, 8, 48, 16, 56, 24, 64, 32,
        39, 7, 47, 15, 55, 23, 63, 31,
        38, 6, 46, 14, 54, 22, 62, 30,
        37, 5, 45, 13, 53, 21, 61, 29,
        36, 4, 44, 12, 52, 20, 60, 28,
        35, 3, 43, 11, 51, 19, 59, 27,
        34, 2, 42, 10, 50, 18, 58, 26,
        33, 1, 41, 9, 49, 17, 57, 25]

# The final permutation is the inverse of the initial permutation; the table is interpreted similarly.

# Expand matrix to get a 48bits matrix of datas to apply the xor with Ki
E = [32, 1, 2, 3, 4, 5,
     4, 5, 6, 7, 8, 9,
     8, 9, 10, 11, 12, 13,
     12, 13, 14, 15, 16, 17,
     16, 17, 18, 19, 20, 21,
     20, 21, 22, 23, 24, 25,
     24, 25, 26, 27, 28, 29,
     28, 29, 30, 31, 32, 1]

# The expansion function is interpreted as for the initial and final permutations. Note that some bits from the input
# are duplicated at the output; e.g. the fifth bit of the input is duplicated in both the sixth and eighth bit of the
# output. Thus, the 32-bit half-block is expanded to 48 bits.

S_BOX = [
    # 1
    [[14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
     [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
     [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
     [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13],
     ],
    # 2
    [[15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
     [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
     [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
     [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9],
     ],
    # 3
    [[10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
     [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
     [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
     [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12],
     ],
    # 4
    [[7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
     [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
     [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
     [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14],
     ],
    # 5
    [[2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
     [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
     [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
     [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3],
     ],
    # 6
    [[12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
     [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
     [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
     [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13],
     ],
    # 7
    [[4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
     [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
     [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
     [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12],
     ],
    # 8
    [[13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
     [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
     [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
     [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11],
     ]
]
# This table lists the eight S-boxes used in DES. Each S-box replaces a 6-bit input with a 4-bit output. Given a 6-bit
# input, the 4-bit output is found by selecting the row using the outer two bits, and the column using the inner four
# bits. For example, an input "011011" has outer bits "01" and inner bits "1101"; noting that the first row is "00"
# and the first column is "0000", the corresponding output for S-box S5 would be "1001" (=9), the value in the second
# row, 14th column.


# Permut made after each SBox substitution for each round
P = [16, 7, 20, 21, 29, 12, 28, 17,
     1, 15, 23, 26, 5, 18, 31, 10,
     2, 8, 24, 14, 32, 27, 3, 9,
     19, 13, 30, 6, 22, 11, 4, 25]
# The P permutation shuffles the bits of a 32-bit half-block.

# Initial permut made on the key
PC_1 = [57, 49, 41, 33, 25, 17, 9,
        1, 58, 50, 42, 34, 26, 18,
        10, 2, 59, 51, 43, 35, 27,
        19, 11, 3, 60, 52, 44, 36,
        63, 55, 47, 39, 31, 23, 15,
        7, 62, 54, 46, 38, 30, 22,
        14, 6, 61, 53, 45, 37, 29,
        21, 13, 5, 28, 20, 12, 4]
# The "Left" and "Right" halves of the table show which bits from the input key form the left and right sections of
# the key schedule state. Note that only 56 bits of the 64 bits of the input are selected; the remaining eight
# (8, 16, 24, 32, 40, 48, 56, 64) were specified for use as parity bits.

# Permut applied on shifted key to get Ki+1
PC_2 = [14, 17, 11, 24, 1, 5, 3, 28,
        15, 6, 21, 10, 23, 19, 12, 4,
        26, 8, 16, 7, 27, 20, 13, 2,
        41, 52, 31, 37, 47, 55, 30, 40,
        51, 45, 33, 48, 44, 49, 39, 56,
        34, 53, 46, 42, 50, 36, 29, 32]

# This permutation selects the 48-bit subkey for each round from the 56-bit key-schedule state.
# This permutation will ignore 8 bits below: Permuted Choice 2 "PC-2" Ignored bits 9,18,22,25,35,38,43,54.

# Matrix that determine the shift for each round of keys
SHIFT = [1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1]


# Bits Rotation
# Before the round sub-key is selected, each half of the key schedule state is rotated left by a number of places.
# This table specifies the number of places rotated.
# The key is divided into two 28-bit parts, Each part is shifted left (circular) one or two bits
# After shifting, two parts are then combined to form a 56 bit temp-key again

# ------------------------------------------FUNCTIONS--------------------------------------------------------------------


def to_bits(s):
    bit_arry = []
    for c in s:
        bits = bin(ord(c))[2:]
        bits = '00000000'[len(bits):] + bits
        bit_arry.extend([int(b) for b in bits])
    return bit_arry


def to_bits_4(s):
    bit_arry = []
    for c in str(s):
        bits = bin(ord(c))[2:]
        bits = '0000'[len(bits):] + bits
        bit_arry.extend([int(b) for b in bits])
    return bit_arry


def from_bits(bits):
    chars = []
    for b in range(len(bits) // 8):
        byte = bits[b * 8:(b + 1) * 8]
        chars.append(chr(int(''.join([str(bit) for bit in byte]), 2)))
    return ''.join(chars)


def check_key(key):  # in bytes 8 * 8 = 64 bits
    if len(key) > 8:
        key = key[:8]
    if len(key) < 8:
        print("key is too short")
        exit()
    return key


def check_text(text):
    if len(text) != 0:
        if len(text) % 8 != 0:
            text = add_padding(text)
        return text
    else:
        exit()


def add_padding(text):  # Data size must be multiple of 8 bytes
    pad_len = 8 - (len(text) % 8)
    for i in range(pad_len):
        text = text + text[i]
    return text


def remove_padding(text, padding):
    print(text)
    print(padding)
    return text[0:-padding]


def splitter_8(string):  # Again 8 bytes => 64 bits
    return [string[k:k + 8] for k in range(0, len(string), 8)]


def splitter_28(string):  # Split for left and right parts
    return [string[k:k + 28] for k in range(0, len(string), 28)]


def splitter_32(string):  # Split for left and right parts
    return [string[k:k + 32] for k in range(0, len(string), 32)]


def splitter_6(string):  # Split for left and right parts
    return [string[k:k + 6] for k in range(0, len(string), 6)]


def shifts(g, d, n):
    return g[n:] + g[:n], d[n:] + d[:n]


def permutation(block, table):
    return [block[x - 1] for x in table]


def permut_key(block, matrix):
    return [block[x - 1] for x in matrix]


def xor(string_1, string_2):
    string_xor = ""
    if len(string_1) > len(string_2):
        for i in range(0, len(string_1)):
            string_xor += str(int(string_1[i]) ^ int(string_2[i]))
    else:
        for i in range(0, len(string_2)):
            string_xor += str(int(string_1[i]) ^ int(string_2[i]))
    return string_xor


def key_generator(password):
    keys = []
    key = to_bits(password)
    key = permutation(key, PC_1)
    left, right = splitter_28(key)
    for i in range(16):
        left, right = shifts(left, right, SHIFT[i])
        merge = left + right
        keys.append(permutation(merge, PC_2))
    return keys


def s_Box(right):
    makro_blocks = splitter_6(right)
    res = list()
    for i in range(len(makro_blocks)):
        block = makro_blocks[i]
        line = int(str(block[0]) + str(block[5]), 2)
        column = int(''.join([str(x) for x in block[1:][:-1]]), 2)
        tmp = S_BOX[i][line][column]
        bin = to_bits_4(tmp)
        res += [int(x) for x in bin]
    return res


def encrypt_des(key, text):
    res = list()
    key = check_key(key)
    if len(text) % 8 != 0:
        text = add_padding(text)
    text = check_text(text)
    keys = key_generator(key)
    text_blocks = splitter_8(text)
    for block in text_blocks:
        block = to_bits(block)
        block = permutation(block, IP)
        left, right = splitter_32(block)
        for i in range(16):
            new_right = permut_key(right, E)
            tmp = xor(keys[i], new_right)
            tmp = s_Box(tmp)
            tmp = permutation(tmp, P)
            tmp = xor(left, tmp)
            left = right
            right = tmp
        res += permutation(right + left, IP_1)
    final_res = from_bits(res)
    return final_res


def decrypt_des(key, text):
    res = list()
    key = check_key(key)
    text = check_text(text)
    keys = key_generator(key)
    text_blocks = splitter_8(text)
    for block in text_blocks:
        block = to_bits(block)
        block = permutation(block, IP)
        left, right = splitter_32(block)
        for i in range(16):
            new_right = permut_key(right, E)
            tmp = xor(keys[15 - i], new_right)
            tmp = s_Box(tmp)
            tmp = permutation(tmp, P)
            tmp = xor(left, tmp)
            left = right
            right = tmp
        res += permutation(right + left, IP_1)
    final_res = from_bits(res)
    return final_res


# -------------------------------------------------GENERATE KEY-------------------------------------------------------
key = generate_key()
# --------------------------------------------------GUI-----------------------------------------------------------------


def encrypt():
    text = e1.get()
    crypto = encrypt_des(key, text)
    message = "Cipher in ASCII: "
    message_ = crypto
    message_2 = "Cipher in utf-8: "
    message_2_ = crypto.encode('utf-8')
    message_3 = "Cipher in base64: "
    message_3_ = base64.standard_b64encode(crypto.encode('utf-8'))
    second = tk.Tk()
    second.geometry('450x150')
    second.title("Encrypted Message")
    tk.Label(second, text=message).grid(row=0, column=1)
    tk.Label(second, text=message_).grid(row=0, column=2)
    tk.Label(second, text=message_2).grid(row=1, column=1)
    tk.Label(second, text=message_2_).grid(row=1, column=2)
    tk.Label(second, text=message_3).grid(row=2, column=1)
    tk.Label(second, text=message_3_).grid(row=2, column=2)


def decrypt():
    text = e2.get()
    message = decrypt_des(key, text)
    second = tk.Tk()
    second.geometry('450x150')
    second.title("Decrypted Message")
    tk.Label(second, text=message).grid(row=0, column=1)


master = tk.Tk()
master.title("DES Cryptographer")
master.geometry('450x150')
tk.Label(master, text="Message to encrypt").grid(row=0)
tk.Label(master, text="Message to decrypt").grid(row=1)
tk.Button(master, text="encrypt", command=encrypt).grid(row=0, column=2)
tk.Button(master, text="decrypt", command=decrypt).grid(row=1, column=2)

e1 = tk.Entry(master)
e2 = tk.Entry(master)

e1.grid(row=0, column=1)
e2.grid(row=1, column=1)

master.mainloop()
