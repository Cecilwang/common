try:
	LL   R2, 0(R1)   ; load linked, R2 gets the current value
	ADDI R3, R2, 1   ; increment, R3 = R2 + 1
	SC   R3, 0(R1)   ; store conditional, try to write back
	BEQ  R3, R0, try ; branch if store fails (R3 == 0)
