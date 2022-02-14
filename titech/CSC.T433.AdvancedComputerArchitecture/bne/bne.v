/******************************************************************************/
/* Simple Processor Ver.07                    2018-12-18  ArchLab. TOKYO TECH */
/* Four stage pipelined processor supporting ADD and BNE,                     */
/* which has no data forwarding                                               */
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
    p.imem.mem[0] = {6'h0, 5'd0, 5'd0, 5'd0, 5'd0, 6'h20};  //   NOP
    p.imem.mem[1] = {6'h0, 5'd5, 5'd1, 5'd5, 5'd0, 6'h20};  // L1:
                                                            //   add  $5, $5, $1
    p.imem.mem[2] = {6'h0, 5'd0, 5'd0, 5'd0, 5'd0, 6'h20};  //   NOP
    p.imem.mem[3] = {6'h0, 5'd0, 5'd0, 5'd0, 5'd0, 6'h20};  //   NOP
    p.imem.mem[4] = {6'h0, 5'd0, 5'd0, 5'd0, 5'd0, 6'h20};  //   NOP
    p.imem.mem[5] = {6'h5, 5'd4, 5'd5, 16'hfffb};           //   bne  $4, $5, L1
    p.imem.mem[6] = {6'h0, 5'd0, 5'd0, 5'd0, 5'd0, 6'h20};  //   NOP
    p.imem.mem[7] = {6'h0, 5'd0, 5'd0, 5'd0, 5'd0, 6'h20};  //   NOP
    p.imem.mem[8] = {6'h0, 5'd0, 5'd0, 5'd5, 5'd0, 6'h20};  //   add  $5, $0, $0
    p.imem.mem[9] = {6'h0, 5'd0, 5'd0, 5'd0, 5'd0, 6'h20};  //   NOP
    p.imem.mem[10]= {6'h0, 5'd0, 5'd0, 5'd0, 5'd0, 6'h20};  //   NOP
    p.imem.mem[11]= {6'h5, 5'd2, 5'd0, 16'hfff5};           //   bne  $2, $0, L1
    p.imem.mem[12]= {6'h0, 5'd0, 5'd0, 5'd0, 5'd0, 6'h20};  //   NOP
    p.imem.mem[13]= {6'h0, 5'd0, 5'd0, 5'd0, 5'd0, 6'h20};  //   NOP

    p.regfile.r[1] = 1;
    p.regfile.r[2] = 22;
    p.regfile.r[3] = 0;
    p.regfile.r[4] = 4;
    p.regfile.r[5] = 0;
  end
  initial #30000 $finish();

  always @(posedge CLK) begin
    if(p.IdEx_OP==6'h5) $write("RRS, RRT, TKN:%8d %8d  %8d\n",
                               p.IdEx_RRS,p.IdEx_RRT,p.Ex_TKN);
  end

  PROCESSOR_07 p(CLK, RST_X);
endmodule

/******************************************************************************/
module PROCESSOR_07(CLK, RST_X);
  input wire CLK, RST_X;

  /***************************************** IF Stage  **********************/
  reg  [31:0] PC;
  wire [31:0] NextPC = Ex_TKN ? IdEx_TPC : PC+4;
  always @(posedge CLK) PC <= #5 (!RST_X) ? 0 : NextPC;

  wire [31:0] If_IR;
  MEM imem(CLK, PC, 32'd0, 1'd0, If_IR);

  reg [31:0] IfId_IR, IfId_NPC;
  always @(posedge CLK) IfId_IR  <= #5 (!RST_X) ? 0 : If_IR;
  always @(posedge CLK) IfId_NPC <= #5 (!RST_X) ? 0 : PC+4;
  /***************************************** ID Stage  **********************/
  wire [5:0]  #10 Id_OP  = IfId_IR[31:26];
  wire [4:0]  #10 Id_RS  = IfId_IR[25:21];
  wire [4:0]  #10 Id_RT  = IfId_IR[20:16];
  wire [4:0]  #10 Id_RD  = IfId_IR[15:11];
  wire [15:0] #10 Id_IMM = IfId_IR[15:0];
  wire [31:0]     Id_BranchAddr = Id_OP == 6'h5 ?
                                  {{16{Id_IMM[15]}}, Id_IMM[15:0]} << 2 : 0;

  wire [31:0] Id_RRS, Id_RRT;
  GPR regfile(CLK, Id_RS, Id_RT, ExWb_RD, ExWb_RSLT, ExWb_OP == 6'h0 ? 1 : 0,
              Id_RRS, Id_RRT);

  reg [5:0]  IdEx_OP;
  reg [31:0] IdEx_RRS, IdEx_RRT, IdEx_TPC;
  reg [4:0]  IdEx_RD;
  always @(posedge CLK) IdEx_OP  <= #5 (!RST_X) ? 0 : Id_OP;
  always @(posedge CLK) IdEx_RRS <= #5 (!RST_X) ? 0 : Id_RRS;
  always @(posedge CLK) IdEx_RRT <= #5 (!RST_X) ? 0 : Id_RRT;
  always @(posedge CLK) IdEx_RD  <= #5 (!RST_X) ? 0 : Id_RD;
  always @(posedge CLK) IdEx_TPC <= #5 (!RST_X) ? 0 : IfId_NPC + Id_BranchAddr;
  /***************************************** EX Stage  **********************/
  wire [31:0] #20 Ex_RSLT = IdEx_RRS + IdEx_RRT;
  wire        #20 Ex_TKN  = (IdEx_OP == 6'h5) ? IdEx_RRS != IdEx_RRT : 0;

  reg [5:0]  ExWb_OP;
  reg [31:0] ExWb_RSLT;
  reg [4:0]  ExWb_RD;
  always @(posedge CLK) ExWb_OP   <= #5 (!RST_X) ? 0 : IdEx_OP;
  // It's unnecessary to assign ExWb_RSLT and ExWb_RD by IdEx_OP because WE0
  // will be set by ExWb_OP properly, but I still do this for a clear wave.
  always @(posedge CLK) ExWb_RSLT <= #5 (!RST_X) ? 0 :
                                           (IdEx_OP == 6'h5) ? 0 : Ex_RSLT;
  always @(posedge CLK) ExWb_RD   <= #5 (!RST_X) ? 0 :
                                           (IdEx_OP == 6'h5) ? 0 : IdEx_RD;
  /***************************************** WB Stage  **********************/

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
  always @(negedge CLK) if(WE0) r[REGNUM2] <= #10 DIN0;
//  always @(posedge CLK) if(WE0) r[REGNUM2] <= #10 DIN0;
endmodule
/******************************************************************************/
