import com.jogamp.opengl.*;
import java.io.*;

public class Shader{
  String vshaderfilename;
  String fshaderfilename;
  String vertexShaderSrc;
  String fragmentShaderSrc;
  int programID;
  
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
