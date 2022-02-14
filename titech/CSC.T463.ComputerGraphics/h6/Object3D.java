import com.jogamp.opengl.*;
import com.jogamp.opengl.util.*; 

abstract class Object3D implements SceneGraphNode{
  public static final int VERTEXPOSITION=0;
  public static final int VERTEXCOLOR=1;
  public static final int VERTEXTEXCOORD0=3;
  public static final int VERTEXNORMAL=2;
  private int[] storedprogramname= new int[1];
  protected PMVMatrix mat;
  protected int programID;

  protected void initCommon(PMVMatrix mat, int programID){
    this.mat = mat;
    this.programID = programID;
  }

  protected void displayCommon(GL2 gl){
    if(programID != 0){
      gl.glUseProgram(programID);
    }
  }

  abstract void init(GL2 gl, PMVMatrix mat, int programID);
  //  abstract void display(GL2 gl, PMVMatrix mats, int programID);
  abstract void display(GL2 gl, PMVMatrix mats);

  void bindProgram(GL2ES2 gl, int programname){
    gl.glGetIntegerv(GL2.GL_CURRENT_PROGRAM, storedprogramname, 0);
    gl.glUseProgram(programname);
  }

  void unbindProgram(GL2ES2 gl){
    gl.glUseProgram(storedprogramname[0]);
  }

}
