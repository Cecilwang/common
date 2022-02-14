import com.jogamp.opengl.*;
import com.jogamp.opengl.util.*;
import java.nio.*;

public class Cube extends Object3D{
  private final float[] VertexData = new float[]{
     1.0f, 1.0f,-1.0f,  0.0f, 0.0f,-1.0f,   1f,1f,0f,1f,   //#0
    -1.0f,-1.0f,-1.0f,  0.0f, 0.0f,-1.0f,   1f,1f,0f,1f,   //#1
    -1.0f, 1.0f,-1.0f,  0.0f, 0.0f,-1.0f,   1f,1f,0f,1f,   //#2
     1.0f, 1.0f,-1.0f,  0.0f, 0.0f,-1.0f,   1f,1f,0f,1f,   //#3
     1.0f,-1.0f,-1.0f,  0.0f, 0.0f,-1.0f,   1f,1f,0f,1f,   //#4
    -1.0f,-1.0f,-1.0f,  0.0f, 0.0f,-1.0f,   1f,1f,0f,1f,   //#5

     1.0f,  1.0f, 1.0f,  1.0f, 0.0f, 0.0f,   0f,1f,1f,1f,  //#6
     1.0f, -1.0f, 1.0f,  1.0f, 0.0f, 0.0f,   0f,1f,1f,1f,  //#7
     1.0f, -1.0f,-1.0f,  1.0f, 0.0f, 0.0f,   0f,1f,1f,1f,  //#8
     1.0f,  1.0f, 1.0f,  1.0f, 0.0f, 0.0f,   0f,1f,1f,1f,  //#9
     1.0f, -1.0f,-1.0f,  1.0f, 0.0f, 0.0f,   0f,1f,1f,1f,  //#10
     1.0f,  1.0f,-1.0f,  1.0f, 0.0f, 0.0f,   0f,1f,1f,1f,  //#11

     1.0f,  1.0f, 1.0f,  0.0f, 1.0f, 0.0f,   1f,0f,1f,1f,  //#12
     1.0f,  1.0f,-1.0f,  0.0f, 1.0f, 0.0f,   1f,0f,1f,1f,  //#13
    -1.0f,  1.0f, 1.0f,  0.0f, 1.0f, 0.0f,   1f,0f,1f,1f,  //#14
     1.0f,  1.0f,-1.0f,  0.0f, 1.0f, 0.0f,   1f,0f,1f,1f,  //#15
    -1.0f,  1.0f,-1.0f,  0.0f, 1.0f, 0.0f,   1f,0f,1f,1f,  //#16
    -1.0f,  1.0f, 1.0f,  0.0f, 1.0f, 0.0f,   1f,0f,1f,1f,  //#17

    -1.0f,  1.0f, 1.0f, -1.0f, 0.0f, 0.0f,   1f,0f,0f,1f,  //#18
    -1.0f,  1.0f,-1.0f, -1.0f, 0.0f, 0.0f,   1f,0f,0f,1f,  //#19
    -1.0f, -1.0f, 1.0f, -1.0f, 0.0f, 0.0f,   1f,0f,0f,1f,  //#20
    -1.0f, -1.0f, 1.0f, -1.0f, 0.0f, 0.0f,   1f,0f,0f,1f,  //#21
    -1.0f,  1.0f,-1.0f, -1.0f, 0.0f, 0.0f,   1f,0f,0f,1f,  //#22
    -1.0f, -1.0f,-1.0f, -1.0f, 0.0f, 0.0f,   1f,0f,0f,1f,  //#23

    -1.0f, -1.0f, 1.0f,  0.0f,-1.0f, 0.0f,   0f,1f,0f,1f,  //#24
    -1.0f, -1.0f,-1.0f,  0.0f,-1.0f, 0.0f,   0f,1f,0f,1f,  //#25
     1.0f, -1.0f, 1.0f,  0.0f,-1.0f, 0.0f,   0f,1f,0f,1f,  //#26
     1.0f, -1.0f, 1.0f,  0.0f,-1.0f, 0.0f,   0f,1f,0f,1f,  //#27
    -1.0f, -1.0f,-1.0f,  0.0f,-1.0f, 0.0f,   0f,1f,0f,1f,  //#28
     1.0f, -1.0f,-1.0f,  0.0f,-1.0f, 0.0f,   0f,1f,0f,1f,  //#29

    -1.0f,  1.0f,1.0f,  0.0f, 0.0f, 1.0f,   0f,0f,1f,1f,   //#30
    -1.0f, -1.0f,1.0f,  0.0f, 0.0f, 1.0f,   0f,0f,1f,1f,   //#31
     1.0f,  1.0f,1.0f,  0.0f, 0.0f, 1.0f,   0f,0f,1f,1f,   //#32
     1.0f, -1.0f,1.0f,  0.0f, 0.0f, 1.0f,   0f,0f,1f,1f,   //#33
     1.0f,  1.0f,1.0f,  0.0f, 0.0f, 1.0f,   0f,0f,1f,1f,   //#34
    -1.0f, -1.0f,1.0f,  0.0f, 0.0f, 1.0f,   0f,0f,1f,1f,   //#35
  };//position            normal              color
  private final int NormalOffset = Float.SIZE/8*3;
  private final int ColorOffset = Float.SIZE/8*6;
  private final int VertexCount = VertexData.length/10;
  private final int VertexSize = VertexData.length*Float.SIZE/8;
  private final FloatBuffer FBVertexData = FloatBuffer.wrap(VertexData);

  private final int[] ElementData = new int[]{
     0, 1, 2, //polygon#0
     3, 4, 5, //pollgon#1
     6, 7, 8, //polygon#2
     9,10,11, //pollgon#3
    12,13,14, //polygon#4
    15,16,17, //pollgon#5
    18,19,20, //polygon#6
    21,22,23, //pollgon#7
    24,25,26, //polygon#8
    27,28,29, //pollgon#9
    30,31,32, //polygon#10
    33,34,35, //pollgon#11
  };
  private final int PolygonCount = ElementData.length/3;
  private final int ElementCount = ElementData.length;
  private final int ElementSize = ElementCount*Integer.SIZE/8;
  private final IntBuffer IBElementData = IntBuffer.wrap(ElementData);
  private int ElementBufferName;
  private int ArrayBufferName;
  private int TextureName;
  private int uniformTexture;

  public Cube(){
  }

  public void init(GL2GL3 gl, PMVMatrix mat, int programID){
    initCommon(mat, programID);
    int[] tmp = new int[1];
    gl.glGenBuffers(1, tmp, 0);
    ArrayBufferName = tmp[0];
    gl.glBindBuffer(GL.GL_ARRAY_BUFFER, ArrayBufferName);
    gl.glBufferData(GL.GL_ARRAY_BUFFER, VertexSize, FBVertexData,
                    GL.GL_STATIC_DRAW);
    gl.glBindBuffer(GL.GL_ARRAY_BUFFER, 0);

    gl.glGenBuffers(1, tmp, 0);
    ElementBufferName = tmp[0];
    gl.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ElementBufferName);
    gl.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, ElementSize, IBElementData,
                    GL.GL_STATIC_DRAW);
    gl.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0);

  }

  public void display(GL2GL3 gl, int programID){
    // If this object has own special shader, bind it
    //bindProgram(gl, ownProgramID);

    gl.glBindBuffer(GL.GL_ARRAY_BUFFER, ArrayBufferName);
    gl.glVertexAttribPointer(VERTEXPOSITION, 3, GL.GL_FLOAT,
                             false, 40, 0);
    gl.glVertexAttribPointer(VERTEXNORMAL, 3, GL.GL_FLOAT,
			     false, 40, NormalOffset);
    gl.glVertexAttribPointer(VERTEXCOLOR, 4, GL.GL_FLOAT,
			     false, 40, ColorOffset);
    gl.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ElementBufferName);

    gl.glEnableVertexAttribArray(VERTEXPOSITION);
    gl.glEnableVertexAttribArray(VERTEXCOLOR);
    gl.glEnableVertexAttribArray(VERTEXNORMAL);

    gl.glDrawElements(GL.GL_TRIANGLES, ElementCount, GL.GL_UNSIGNED_INT, 0);

    gl.glDisableVertexAttribArray(VERTEXPOSITION);
    gl.glDisableVertexAttribArray(VERTEXNORMAL);
    gl.glDisableVertexAttribArray(VERTEXCOLOR);

    gl.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0);
    gl.glBindBuffer(GL.GL_ARRAY_BUFFER, 0);
   // Unbind the shader program of this object and restore the shader of the
   // parent object
   //  unbindProgram(gl);

    /* // drawing this object without shader
    gl.glUseProgram(0);
    gl.glActiveTexture(GL.GL_TEXTURE0);
    gl.glBindBuffer(GL.GL_ARRAY_BUFFER, ArrayBufferName);
    gl.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ElementBufferName);
    gl.glEnableClientState(GL2.GL_VERTEX_ARRAY);
    gl.glEnableClientState(GL2.GL_NORMAL_ARRAY);
    gl.glEnableClientState(GL2.GL_COLOR_ARRAY);
    gl.glVertexPointer(3, GL.GL_FLOAT, 48, 0);
    gl.glNormalPointer(GL.GL_FLOAT, 48, NormalOffset);
    gl.glColorPointer(4, GL.GL_FLOAT, 48, ColorOffset);
    gl.glDrawElements(GL.GL_TRIANGLES, ElementCount, GL.GL_UNSIGNED_INT, 0);
    //gl.glDrawArrays(GL.GL_TRIANGLES,0,3);
    */

    /* // drawing a polygon by the most traditional way
    gl.glUseProgram(0);
    gl.glBegin(GL2.GL_TRIANGLES);
    gl.glColor3f(1f,0f,1f);
    gl.glVertex3f(-0.5f,-0.5f,-1f);
    gl.glColor3f(1f,1f,0f);
    gl.glVertex3f(0.5f,-0.5f,-1f);
    gl.glColor3f(0f,1f,1f);
    gl.glVertex3f(0.5f,0.5f,-1f);
    gl.glEnd();
    */
  }
}
