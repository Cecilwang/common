/******************************************************************************/
/* Simple Processor Ver.01                    2018-12-05  ArchLab. TOKYO TECH */
/* single-cycle processor supporting ADD                                      */
/******************************************************************************/
`default_nettype none
`timescale 1ns/100ps
/******************************************************************************/
module top();
  reg CLK, RST_X;
  initial begin CLK = 1; forever #50 CLK = ~CLK; end
  initial begin RST_X = 0; #140 RST_X = 1;       end
  initial begin #600 $finish();                 end
  initial begin $dumpfile("wave.vcd"); $dumpvars(0, p); end

  initial begin /* initialize the instruction memory */
    p.imem.mem[0] = {6'h0, 5'd0, 5'd0, 5'd0, 5'd0, 6'h20};
    p.imem.mem[1] = {6'h8, 5'd0, 5'd8, 16'd3};
    p.imem.mem[2] = {6'h8, 5'd0, 5'd9, 16'd5};
    p.imem.mem[3] = {6'h0, 5'd8, 5'd9, 5'd10, 5'd0, 6'h20};
  end

  initial begin /* initialize the register file */
    p.regfile.r[1] = 22;
    p.regfile.r[2] = 33;
  end

  always @(posedge CLK) begin
    $write("%d: %d %x %x: %d %d %d\n",
           $stime, RST_X, p.pc, p.ir, p.rrs, p.rrt, p.result);
  end

  PROCESSOR_01 p(CLK, RST_X);
endmodule

/******************************************************************************/
module PROCESSOR_01(CLK, RST_X);
  input wire CLK, RST_X;

  reg  [31:0] pc;
  wire [31:0] ir;
  wire [31:0] rrs, rrt, result;

  always @(posedge CLK) pc <= #5 (!RST_X) ? 0 : pc + 4;

  IMEM imem(CLK, pc, ir); /* instruction memory */

  wire [5:0]   #10 op    = ir[31:26];
  wire [4:0]   #10 rs    = ir[25:21];
  wire [4:0]   #10 rt    = ir[20:16];
  wire [4:0]   #10 rd    = ir[15:11];
  wire [15:0]  #10 imm16 = ir[15:0];
  wire [31:0]  #10 imm32 = {{16{imm16[15]}}, imm16[15:0]};
  wire [31:0]  #10 rt_rd    = (op == 6'h0) ? rd : rt;
  wire [31:0]  #10 rrt_imm  = (op == 6'h0) ? rrt : imm32;
  assign #20 result = rrs + rrt_imm;
  GPR regfile(CLK, rs, rt, rt_rd, result, 1, rrs, rrt); /* register file */

endmodule

/******************************************************************************/
module IMEM(CLK, ADDR, D_OUT); /* Instruction Memory */
  input  wire        CLK;
  input  wire [31:0] ADDR;
  output wire [31:0] D_OUT;

  reg [31:0] mem[0:1024*8-1]; /* 8K word memory */
  assign #15 D_OUT = mem[ADDR[14:2]];
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
