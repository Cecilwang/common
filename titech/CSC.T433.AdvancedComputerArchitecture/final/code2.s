addi $s1, $zero, 0         # set A's address
addi $t0, $zero, 200       # 200
add  $t1, $zero, $zero     # i = 0
nop
add  $t2, $zero, $s1       # copy A's address
nop
L1:
beq  $t1, $t0, END1        # if i == 200 break
nop
nop
sw   $t1, 0($t2)           # A[i] = i
addi $t2, $t2, 4           # move to A[i+1]
addi $t1, $t1, 1           # i += 1
j    L1
nop
nop
END1:
add  $t2, $zero, $s1       # copy A's address
nop
nop
nop
addi $t0, $t2, 796         # A[199]
nop
nop
nop
L2:
beq  $t2, $t0, END2        # if at A[199] break
nop
nop
lw   $t3, 0($t2)           # load A[i-1]
lw   $t4, 4($t2)           # load A[i]
nop
nop
nop
add  $t3, $t3, $t4         # $t3 = A[i-1] + A[i]
nop
nop
nop
sw   $t3, 4($t2)           # A[i] = $t3
addi $t2, $t2, 4           # move to A[i]
j    L2
nop
nop
END2:
add  $t2, $zero, $s1       # copy A's address
add  $s0, $zero, $zero     # sum = 0
nop
nop
addi $t0, $t2, 800         # A[200]
nop
nop
nop
L3:
beq  $t2, $t0, END3        # if at A[200] break
nop
nop
lw   $t3, 0($t2)           # load A[i]
nop
nop
nop
add  $s0, $s0, $t3         # sum += A[i]
addi $t2, $t2, 4           # move to A[i+1]
j    L3
nop
nop
END3:
