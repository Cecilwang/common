/******************************************************************************/
/* Simple Processor Ver.06                    2018-12-15  ArchLab. TOKYO TECH */
/* Four stage pipelined processor supporting ADD, which has no data forwarding*/
/******************************************************************************/
`default_nettype none
`timescale 1ns/100ps
/******************************************************************************/
module top();
  reg CLK, RST_X;
  initial begin CLK = 1; forever #50 CLK = ~CLK; end
  initial begin RST_X = 0; #140 RST_X = 1;       end
  initial begin $dumpfile("wave.vcd"); $dumpvars(0, p); end

  initial begin /* initialize the instruction & data memory  & regfile */
    p.imem.mem[0] = {6'h0, 5'd0, 5'd0, 5'd0, 5'd0, 6'h20};
    p.imem.mem[1] = {6'h0, 5'd0, 5'd0, 5'd0, 5'd0, 6'h20};
    p.imem.mem[2] = {6'h0, 5'd1, 5'd1, 5'd1, 5'd0, 6'h20};
    p.imem.mem[3] = {6'h0, 5'd2, 5'd2, 5'd2, 5'd0, 6'h20};
    p.imem.mem[4] = {6'h0, 5'd3, 5'd3, 5'd3, 5'd0, 6'h20};
    p.imem.mem[5] = {6'h0, 5'd4, 5'd4, 5'd4, 5'd0, 6'h20};
    p.regfile.r[1] = 22;
    p.regfile.r[2] = 33;
    p.regfile.r[3] = 44;
    p.regfile.r[4] = 55;
  end
  initial #900 begin
    $write("r[1] = %d  # this must be 44\n",  p.regfile.r[1]);
    $write("r[2] = %d  # this must be 66\n",  p.regfile.r[2]);
    $write("r[3] = %d  # this must be 88\n",  p.regfile.r[3]);
    $write("r[4] = %d  # this must be 110\n", p.regfile.r[4]);
    $finish();
  end

  PROCESSOR_06 p(CLK, RST_X);
endmodule

/******************************************************************************/
module PROCESSOR_06(CLK, RST_X);
  input wire CLK, RST_X;

  /***************************************** IF Stage  **********************/
  reg  [31:0] PC;
  always @(posedge CLK) PC <= #5 (!RST_X) ? 0 : PC + 8;

  wire [31:0] If_IR1, If_IR2;
  MEM imem(CLK, PC,   32'd0, 1'd0, If_IR1, If_IR2);

  reg [31:0] IfId_IR1, IfId_IR2;
  always @(posedge CLK) IfId_IR1 <= #5 (!RST_X) ? 0 : If_IR1;
  always @(posedge CLK) IfId_IR2 <= #5 (!RST_X) ? 0 : If_IR2;

  /***************************************** ID Stage  **********************/
  wire [4:0]  #10 Id_RS1 = IfId_IR1[25:21];
  wire [4:0]  #10 Id_RT1 = IfId_IR1[20:16];
  wire [4:0]  #10 Id_RD1 = IfId_IR1[15:11];
  wire [4:0]  #10 Id_RS2 = IfId_IR2[25:21];
  wire [4:0]  #10 Id_RT2 = IfId_IR2[20:16];
  wire [4:0]  #10 Id_RD2 = IfId_IR2[15:11];

  wire [31:0] Id_RRS1, Id_RRT1, Id_RRS2, Id_RRT2;
  GPR regfile(CLK, Id_RS1, Id_RT1, ExWb_RD1, ExWb_RSLT1, 1, Id_RRS1, Id_RRT1, 
	           Id_RS2, Id_RT2, ExWb_RD2, ExWb_RSLT2, 1, Id_RRS2, Id_RRT2);

  reg [31:0] IdEx_RRS1, IdEx_RRT1, IdEx_RRS2, IdEx_RRT2;
  reg [4:0]  IdEx_RD1, IdEx_RD2;
  always @(posedge CLK) IdEx_RRS1 <= #5 (!RST_X) ? 0 : Id_RRS1;
  always @(posedge CLK) IdEx_RRT1 <= #5 (!RST_X) ? 0 : Id_RRT1;
  always @(posedge CLK) IdEx_RD1  <= #5 (!RST_X) ? 0 : Id_RD1;
  always @(posedge CLK) IdEx_RRS2 <= #5 (!RST_X) ? 0 : Id_RRS2;
  always @(posedge CLK) IdEx_RRT2 <= #5 (!RST_X) ? 0 : Id_RRT2;
  always @(posedge CLK) IdEx_RD2  <= #5 (!RST_X) ? 0 : Id_RD2;

  /***************************************** EX Stage  **********************/
  wire [31:0] #20 Ex_RSLT1 = IdEx_RRS1 + IdEx_RRT1;
  wire [31:0] #20 Ex_RSLT2 = IdEx_RRS2 + IdEx_RRT2;

  reg [31:0] ExWb_RSLT1, ExWb_RSLT2;
  reg [4:0]  ExWb_RD1, ExWb_RD2;
  always @(posedge CLK) ExWb_RSLT1 <= #5 (!RST_X) ? 0 : Ex_RSLT1;
  always @(posedge CLK) ExWb_RD1   <= #5 (!RST_X) ? 0 : IdEx_RD1;
  always @(posedge CLK) ExWb_RSLT2 <= #5 (!RST_X) ? 0 : Ex_RSLT2;
  always @(posedge CLK) ExWb_RD2   <= #5 (!RST_X) ? 0 : IdEx_RD2;

  /***************************************** WB Stage  **********************/
endmodule

/******************************************************************************/
module MEM(CLK, ADDR, D_IN, D_WE, D_OUT1, D_OUT2); /*Instruction & Data Memory*/
  input  wire        CLK;
  input  wire [31:0] ADDR, D_IN;
  input  wire        D_WE;
  output wire [31:0] D_OUT1, D_OUT2;

  reg [31:0] mem[0:1024*8-1]; /* 8K word memory */
  assign #15 D_OUT1 = mem[ADDR[14:2]];
  assign #15 D_OUT2 = mem[ADDR[14:2]+1];
  always @(posedge CLK) if(D_WE) mem[ADDR[14:2]] <= #10 D_IN;
endmodule

/* 32bitx32 2R/1W General Purpose Registers (Register File)                   */
/******************************************************************************/
module GPR(CLK, REGNUM0_0, REGNUM0_1, REGNUM0_2, DIN0, WE0, DOUT0_0, DOUT0_1, 
	        REGNUM1_0, REGNUM1_1, REGNUM1_2, DIN1, WE1, DOUT1_0, DOUT1_1);
  input  wire        CLK;
  input  wire  [4:0] REGNUM0_0, REGNUM0_1, REGNUM0_2;
  input  wire  [4:0] REGNUM1_0, REGNUM1_1, REGNUM1_2;
  input  wire [31:0] DIN0, DIN1;
  input  wire        WE0, WE1;
  output wire [31:0] DOUT0_0, DOUT0_1, DOUT1_0, DOUT1_1;

  reg [31:0] r[0:31];
  assign #15 DOUT0_0 = (REGNUM0_0==0) ? 0 : r[REGNUM0_0];
  assign #15 DOUT0_1 = (REGNUM0_1==0) ? 0 : r[REGNUM0_1];
  always @(negedge CLK) if(WE0) r[REGNUM0_2] <= #10 DIN0;
  assign #15 DOUT1_0 = (REGNUM1_0==0) ? 0 : r[REGNUM1_0];
  assign #15 DOUT1_1 = (REGNUM1_1==0) ? 0 : r[REGNUM1_1];
  always @(negedge CLK) if(WE1) r[REGNUM1_2] <= #10 DIN1;
endmodule
/******************************************************************************/
