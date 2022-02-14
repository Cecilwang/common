import com.jogamp.opengl.*;
import com.jogamp.opengl.util.*;

abstract class Object3D{
  public static final int VERTEXPOSITION=0;
  public static final int VERTEXCOLOR=1;
  public static final int VERTEXTEXCOORD0=3;
  public static final int VERTEXNORMAL=2;
  public static final int VERTEXTANGENT=4;
  private int[] storedprogramID = new int[1];
  protected PMVMatrix mat;
  protected Shader shader;

  protected void initCommon(PMVMatrix mat, Shader shader){
    this.mat = mat;
    if(shader != null){
      this.shader = shader;
    }
  }

  abstract void init(GL2 gl, PMVMatrix mat, Shader shader);
  abstract void display(GL2 gl, PMVMatrix mats, Vec3 lightpos, Vec3 lightcolor);

  void bindProgram(GL2ES2 gl, Shader shader){
    gl.glGetIntegerv(GL2.GL_CURRENT_PROGRAM, storedprogramID, 0);
    gl.glUseProgram(shader.getID());
  }

  void unbindProgram(GL2ES2 gl){
    gl.glUseProgram(storedprogramID[0]);
  }
}
