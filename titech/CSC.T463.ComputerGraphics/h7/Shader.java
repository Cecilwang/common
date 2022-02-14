import com.jogamp.opengl.*;
import com.jogamp.opengl.util.*;
import java.io.*;

public class Shader{
  String vshaderfilename;
  String fshaderfilename;
  String vertexShaderSrc;
  String fragmentShaderSrc;
  int programID;
  int uniformMat;
  int uniformLightPos;
  int uniformLightColor;

  public Shader(String vshaderfilename, String fshaderfilename){
    this.vshaderfilename = new String(vshaderfilename);
    this.fshaderfilename = new String(fshaderfilename);
    vertexShaderSrc = loadFile(vshaderfilename);
    fragmentShaderSrc = loadFile(fshaderfilename);
  }

  public void init(GL2ES2 gl){
    int shaderid;
    int[] srcLength={0};
    String[] src = {""};

    programID = gl.glCreateProgram();
    shaderid = gl.glCreateShader(GL2GL3.GL_VERTEX_SHADER);
    src[0] = vertexShaderSrc;
    srcLength[0] = vertexShaderSrc.length();
    gl.glShaderSource(shaderid, 1, src, srcLength, 0);
    gl.glCompileShader(shaderid);
    checkCompileError(gl, shaderid, vshaderfilename);
    gl.glAttachShader(programID, shaderid); 
    gl.glDeleteShader(shaderid);
    vertexShaderSrc = null;

    shaderid = gl.glCreateShader(GL2GL3.GL_FRAGMENT_SHADER);
    src[0] = fragmentShaderSrc;
    srcLength[0] = fragmentShaderSrc.length();
    gl.glShaderSource(shaderid, 1, src, srcLength, 0);
    gl.glCompileShader(shaderid);
    checkCompileError(gl, shaderid, fshaderfilename);
    gl.glAttachShader(programID, shaderid); 
    gl.glDeleteShader(shaderid);
    fragmentShaderSrc = null;
    gl.glBindAttribLocation(programID,Object3D.VERTEXPOSITION, "inposition");
    gl.glBindAttribLocation(programID,Object3D.VERTEXCOLOR, "incolor");
    gl.glBindAttribLocation(programID,Object3D.VERTEXNORMAL, "innormal");
    gl.glBindAttribLocation(programID,Object3D.VERTEXTEXCOORD0,"intexcoord0");
    gl.glBindAttribLocation(programID,Object3D.VERTEXTANGENT,"intangent");
    link(gl);
  } 

  public void link(GL2ES2 gl){
    int[] linkStatus = new int[1];
    gl.glLinkProgram(programID);
    gl.glGetProgramiv(programID, GL2ES2.GL_LINK_STATUS, linkStatus, 0);
    if(linkStatus[0] == GL.GL_FALSE){
      final int logMax = 8192;
      byte[] log = new byte[logMax];
      int[] logLength = new int[1];
      System.err.println("link error");
      gl.glGetProgramInfoLog(programID, logMax, logLength, 0, log, 0);
      showLog(log, logLength[0]);
      System.exit(1);
    }
    uniformMat = gl.glGetUniformLocation(programID, "mat");
    uniformLightPos = gl.glGetUniformLocation(programID, "lightpos");
    uniformLightColor = gl.glGetUniformLocation(programID, "lightcolor");
  }

  private String loadFile(String fname){
    BufferedReader brv = null;
    try{
      brv = new BufferedReader(new FileReader(fname));
    } catch(FileNotFoundException e){
      System.out.println(fname + " not found");
      System.exit(1);
    }
    String ret = new String();
    try{
      String line;
      while ((line=brv.readLine()) != null){
        ret += line + "\n";
      }
    }catch(IOException e){
      System.out.println(e);
      System.exit(1);
    }
    try{
      brv.close();
    }catch(IOException ex){
    }
    return ret;
  }
    
  private void checkCompileError(GL2ES2 gl, int shaderid, String fname){
    int[] compileStatus = new int[1];
    gl.glGetShaderiv(shaderid, GL2ES2.GL_COMPILE_STATUS, compileStatus, 0);
    if(compileStatus[0] == GL.GL_FALSE){
      System.err.println("compile error in shader source of "+fname);
      final int logMax = 8192;
      byte[] log = new byte[logMax];
      int[] logLength = new int[1];
      gl.glGetShaderInfoLog(shaderid, logMax, logLength, 0, log, 0);
      showLog(log, logLength[0]);
      System.exit(1);
    }
  }

  public int getID(){
    return programID;
  }

  public void setMatrixAndLight(GL2ES2 gl, PMVMatrix mats, Vec3 lightpos,
                                Vec3 lightcolor){
    gl.glUniformMatrix4fv(uniformMat, 4, false, mats.glGetPMvMvitMatrixf());
    gl.glUniform3f(uniformLightPos,
                   lightpos.data[0], lightpos.data[1], lightpos.data[2]);
    gl.glUniform3f(uniformLightColor,
                   lightcolor.data[0], lightcolor.data[1], lightcolor.data[2]);
  }
  
  public void validateProgram(GL2ES2 gl){
    int[] validateStatus = new int[1];
    gl.glValidateProgram(programID);
    gl.glGetProgramiv(programID, GL2ES2.GL_VALIDATE_STATUS, validateStatus, 0);
    if(validateStatus[0] == GL.GL_FALSE){
      final int logMax = 8192;
      byte[] log = new byte[logMax];
      int[] logLength = new int[1];      
      System.err.println("validate error");
      gl.glGetProgramInfoLog(programID, logMax, logLength, 0, log, 0);
      showLog(log, logLength[0]);
      System.exit(1);
    }
  }

  private void showLog(byte[] log, int length){
    for(int i = 0; i < length; i++){
      System.err.print((char)log[i]);
    }
  }

}
