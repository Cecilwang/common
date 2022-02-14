/******************************************************************************/
/* Simple Processor Ver.07                    2018-12-18  ArchLab. TOKYO TECH */
/* Four stage pipelined processor supporting ADD and BNE,
/* which has no data forwarding      */
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
    p.imem.mem[0] = {6'h0, 5'd0, 5'd0, 5'd16, 5'd0, 6'h20};
    p.imem.mem[1] = {6'h8, 5'd0, 5'd8, 16'd101};
    p.imem.mem[2] = {6'h0, 5'd0, 5'd0, 5'd9, 5'd0, 6'h20};
    p.imem.mem[3] = {6'h0, 5'd0, 5'd0, 5'd0, 5'd0, 6'h20};
    p.imem.mem[4] = {6'h0, 5'd0, 5'd0, 5'd0, 5'd0, 6'h20};
    p.imem.mem[5] = {6'h0, 5'd0, 5'd0, 5'd0, 5'd0, 6'h20};
    // L1
    p.imem.mem[6] = {6'h4, 5'd8, 5'd9, 16'h16};
    p.imem.mem[7] = {6'h0, 5'd0, 5'd0, 5'd0, 5'd0, 6'h20};
    p.imem.mem[8] = {6'h0, 5'd0, 5'd0, 5'd0, 5'd0, 6'h20};
    p.imem.mem[9] = {6'h0, 5'd0, 5'd0, 5'd10, 5'd0, 6'h20};
    p.imem.mem[10] = {6'h0, 5'd0, 5'd0, 5'd0, 5'd0, 6'h20};
    p.imem.mem[11] = {6'h0, 5'd0, 5'd0, 5'd0, 5'd0, 6'h20};
    p.imem.mem[12] = {6'h0, 5'd0, 5'd0, 5'd0, 5'd0, 6'h20};
    // L2
    p.imem.mem[13] = {6'h4, 5'd8, 5'd10, 16'hb};
    p.imem.mem[14] = {6'h0, 5'd0, 5'd0, 5'd0, 5'd0, 6'h20};
    p.imem.mem[15] = {6'h0, 5'd0, 5'd0, 5'd0, 5'd0, 6'h20};
    p.imem.mem[16] = {6'h0, 5'd9, 5'd10, 5'd11, 5'd0, 6'h20};
    p.imem.mem[17] = {6'h0, 5'd0, 5'd0, 5'd0, 5'd0, 6'h20};
    p.imem.mem[18] = {6'h0, 5'd0, 5'd0, 5'd0, 5'd0, 6'h20};
    p.imem.mem[19] = {6'h0, 5'd0, 5'd0, 5'd0, 5'd0, 6'h20};
    p.imem.mem[20] = {6'h0, 5'd16, 5'd11, 5'd16, 5'd0, 6'h20};
    p.imem.mem[21] = {6'h8, 5'd10, 5'd10, 16'd1};
    p.imem.mem[22] = {6'h2, 26'hd};
    p.imem.mem[23] = {6'h0, 5'd0, 5'd0, 5'd0, 5'd0, 6'h20};
    p.imem.mem[24] = {6'h0, 5'd0, 5'd0, 5'd0, 5'd0, 6'h20};
    // END2
    p.imem.mem[25] = {6'h8, 5'd9, 5'd9, 16'd1};
    p.imem.mem[26] = {6'h2, 26'h6};
    p.imem.mem[27] = {6'h0, 5'd0, 5'd0, 5'd0, 5'd0, 6'h20};
    p.imem.mem[28] = {6'h0, 5'd0, 5'd0, 5'd0, 5'd0, 6'h20};
    // END1
    p.imem.mem[29] = {6'h0, 5'd16, 5'd0, 5'd16, 5'd0, 6'h20}; // show s0
  end
  initial #30000 $finish();

  PROCESSOR_07 p(CLK, RST_X);
endmodule

/******************************************************************************/
module PROCESSOR_07(CLK, RST_X);
  input wire CLK, RST_X;

  /***************************************** IF Stage  **********************/
  reg  [31:0] PC;
  wire [31:0] #20 If_NPC = PC + 4;
  always @(posedge CLK) PC <= #5 (!RST_X) ? 0 : (Ex_TKN) ? IdEx_TPC : If_NPC;

  wire [31:0] If_IR;
  MEM imem(CLK, PC, 32'd0, 1'd0, If_IR);

  reg [31:0] IfId_IR, IfId_PC, IfId_NPC;
  always @(posedge CLK) IfId_IR  <= #5 (!RST_X) ? 0 : If_IR;
  always @(posedge CLK) IfId_PC  <= #5 (!RST_X) ? 0 : PC;
  always @(posedge CLK) IfId_NPC <= #5 (!RST_X) ? 0 : If_NPC;
  /***************************************** ID Stage  **********************/
  wire [5:0]  #10 Id_OP = IfId_IR[31:26];
  wire [4:0]  #10 Id_RS = IfId_IR[25:21];
  wire [4:0]  #10 Id_RT = IfId_IR[20:16];
  wire [4:0]  #10 Id_RD = IfId_IR[15:11];
  wire [15:0] #10 Id_IM = IfId_IR[15:0];
  wire [15:0] #10 Id_address = IfId_IR[25:0];

  wire [31:0] #10 Id_SignExtImm = {{16{Id_IM[15]}}, Id_IM[15:0]};
  wire [31:0] #10 Id_JumpAddr   = {IfId_PC[31:28], Id_address, 2'h0};
  wire [31:0] #10 Id_BranchAddr = {{14{Id_IM[15]}}, Id_IM[15:0], 2'h0};

  wire [4:0]  #20 Id_RDST = (Id_OP==6'h23 || Id_OP==6'h8) ? Id_RT : Id_RD;
  wire [31:0] #20 Id_TPC = (Id_OP==6'h2) ? Id_JumpAddr : IfId_NPC+Id_BranchAddr;

  wire [31:0] Id_RRS, Id_RRT;
  GPR regfile(CLK, Id_RS, Id_RT, MaWb_RDST, MaWb_RSLT, MaWb_We, Id_RRS, Id_RRT);

  reg [31:0] IdEx_RRS, IdEx_rrt, IdEx_RRT, IdEx_TPC;
  reg [4:0]  IdEx_RDST;
  reg [5:0]  IdEx_OP;
  always @(posedge CLK) begin
    IdEx_OP   <= #5 (!RST_X) ? 0 : Id_OP;
    IdEx_RRS  <= #5 (!RST_X) ? 0 : Id_RRS;
    IdEx_rrt  <= #5 (!RST_X) ? 0 : Id_RRT;
    IdEx_RRT  <= #5 (!RST_X) ? 0 : (Id_OP==6'h8 || Id_OP==6'h23 || Id_OP==6'h2b)
                                   ? Id_SignExtImm : Id_RRT;
    IdEx_RDST <= #5 (!RST_X) ? 0 : Id_RDST;
    IdEx_TPC  <= #5 (!RST_X) ? 0 : Id_TPC;
  end
  /***************************************** EX Stage  **********************/
  wire [31:0] #20 Ex_RSLT = IdEx_RRS + IdEx_RRT;
  wire        #20 Ex_TKN  = ((IdEx_OP==6'h4 && (IdEx_RRS==IdEx_RRT)) ||
                             (IdEx_OP==6'h5 && (IdEx_RRS!=IdEx_RRT)) ||
                             (IdEx_OP==6'h2));

  reg [31:0] ExMa_rrt, ExMa_RSLT;
  reg [5:0]  ExMa_OP;
  reg [4:0]  ExMa_RDST;
  always @(posedge CLK) ExMa_rrt <= #5 (!RST_X) ? 0 : IdEx_rrt;
  always @(posedge CLK) ExMa_RSLT <= #5 (!RST_X) ? 0 : Ex_RSLT;
  always @(posedge CLK) ExMa_OP   <= #5 (!RST_X) ? 0 : IdEx_OP;
  always @(posedge CLK) ExMa_RDST <= #5 (!RST_X) ? 0 : IdEx_RDST;
  /***************************************** MA Stage  **********************/
  wire [31:0] Ma_dmout;
  MEM dm(CLK, ExMa_RSLT, ExMa_rrt, ExMa_OP==6'h2b ? 1'd1: 1'd0, Ma_dmout);

  wire WE = (ExMa_OP == 0 || ExMa_OP == 6'h8 || ExMa_OP == 6'h23) ? 1 : 0;

  reg [31:0] MaWb_RSLT;
  reg        MaWb_We;
  reg [4:0]  MaWb_RDST;
  always @(posedge CLK) MaWb_RSLT <= #5 (!RST_X) ? 0 :
                                      ((ExMa_OP==6'h23) ? Ma_dmout : ExMa_RSLT);
  always @(posedge CLK) MaWb_We   <= #5 (RST_X && WE) ? 1 : 0;
  always @(posedge CLK) MaWb_RDST <= #5 (!RST_X) ? 0 : ExMa_RDST;
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
