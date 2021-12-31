/******************************************************************************/
/* Simple Processor Ver.03                    2018-12-09  ArchLab. TOKYO TECH */
/* single-cycle processor supporting ADD, ADDI and LW                         */
/******************************************************************************/
`default_nettype none
`timescale 1ns/100ps
/******************************************************************************/
module top();
  reg CLK, RST_X;
  initial begin CLK = 1; forever #50 CLK = ~CLK; end
  initial begin RST_X = 0; #140 RST_X = 1;       end
  initial begin #700 $finish();                 end
  initial begin $dumpfile("wave.vcd"); $dumpvars(0, p); end

  initial begin /* initialize the instruction & data memory */
    p.imem.mem[0] = {6'h0, 5'd0, 5'd0, 5'd0, 5'd0, 6'h20};
    p.imem.mem[1] = {6'h8, 5'd0, 5'd8, 16'd8};   // I_Format: addi $t0, $zero, 8
    p.imem.mem[2] = {6'h2b,5'd8, 5'd8, 16'd4};   // I_Format: sw   $t0, 4($t0)
    p.imem.mem[3] = {6'h23,5'd8, 5'd9, 16'd4};   // I_Format: lw   $t1, 4($t0)
    p.imem.mem[4] = {6'h8, 5'd9, 5'd10,16'h6};   // R_Format: addi $t2, $t1, 6
  end

  always @(posedge CLK) begin
    $write("%d: %d %x %x: %d %d %d %d\n",
           $stime, RST_X, p.pc, p.ir, p.rrs, p.RRT, p.rslt, p.wrslt);
  end

  PROCESSOR_03 p(CLK, RST_X);
endmodule

/******************************************************************************/
module PROCESSOR_03(CLK, RST_X);
  input wire CLK, RST_X;

  reg  [31:0] pc;
  wire [31:0] ir;
  wire [31:0] rrs, rrt;

  always @(posedge CLK) pc <= #5 (!RST_X) ? 0 : pc + 4;

  MEM imem(CLK, pc, 32'd0, 1'd0, ir);                   /* instruction memory */

  wire [5:0]  #10 op = ir[31:26];
  wire [4:0]  #10 rs = ir[25:21];
  wire [4:0]  #10 rt = ir[20:16];
  wire [4:0]  #10 rd = ir[15:11];
  wire        #10 i_form = (op==6'h8 || op==6'h23 || op==6'h2b); /* I-format */

  wire [4:0]  #10 rdst  = (i_form) ? rt : rd;                /* MUX_1 */
  wire signed [31:0] #10 imm = {{16{ir[15]}}, ir[15:0]};     /* 32 immediate */

  GPR regfile(CLK, rs, rt, rdst, wrslt, op==6'h2b ? 1'd0: 1'd1, rrs, rrt);

  wire signed [31:0] #10 RRT = (i_form) ? imm : rrt;         /* MUX_2 */
  wire [31:0] #20 rslt = rrs + RRT;                          /* ALU   */

  wire [31:0] dmout;
  MEM dm(CLK, rslt, rrt, op==6'h2b ? 1'd1: 1'd0, dmout);     /* data memory */
  wire [31:0] #10 wrslt = (op==6'h23) ? dmout : rslt;        /* MUX_3 */
endmodule

/******************************************************************************/
module MEM(CLK, ADDR, D_IN, D_WE, D_OUT); /* Instruction & Data Memory */
  input  wire        CLK;
  input  wire [31:0] ADDR, D_IN;
  input  wire        D_WE;
  output wire [31:0] D_OUT;

  reg [31:0] mem[0:1024*8-1]; /* 8K word memory */
  assign #15 D_OUT = mem[ADDR[14:2]];
  always @(posedge CLK) if(D_WE) mem[ADDR[14:2]] <= #10 D_IN;
endmodule

/* 32bitx32 2R/1W General Purpose Registers (Register File)                   */
/******************************************************************************/
module GPR(CLK, REGNUM0, REGNUM1, REGNUM2, DIN0, WE0, DOUT0, DOUT1);
  input  wire        CLK;
  input  wire  [4:0] REGNUM0, REGNUM1, REGNUM2;
  input  wire [31:0] DIN0;
  input  wire        WE0;
  output wire [31:0] DOUT0, DOUT1;

  reg [31:0] r[0:31];
  assign #15 DOUT0 = (REGNUM0==0) ? 0 : r[REGNUM0];
  assign #15 DOUT1 = (REGNUM1==0) ? 0 : r[REGNUM1];
  always @(posedge CLK) if(WE0) r[REGNUM2] <= #10 DIN0;
endmodule
/******************************************************************************/
