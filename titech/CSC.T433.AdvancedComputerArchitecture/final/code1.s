add  $s0, $zero, $zero  # sum = 0
addi $t0, $zero, 101    # 101
add  $t1, $zero, $zero  # i = 0
nop                     # waiting until $t0/$t1 is written back
nop
nop
L1:
beq  $t1, $t0, END1     # if i == 101 break
nop                     # waiting until beq execution
nop
add  $t2, $zero, $zero  # j = 0
nop                     # waiting until $t2 is written back
nop
nop
L2:
beq  $t2, $t0, END2     # if j == 101 break
nop                     # waiting until beq execution
nop
add  $t3, $t1, $t2      # k = i + j
nop                     # waiting for k
nop
nop
add  $s0, $s0, $t3      # sum += k
addi $t2, $t2, 1        # j += 1
j    L2
nop                     # waiting until j execution
nop
END2:
addi $t1, $t1, 1        # i += 1
j    L1                 # waiting until j execution
nop
nop
END1:
