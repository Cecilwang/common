import com.jogamp.newt.event.KeyEvent;
import com.jogamp.newt.event.KeyListener;
import com.jogamp.newt.event.MouseEvent;
import com.jogamp.newt.event.MouseListener;
import com.jogamp.opengl.*;
import com.jogamp.opengl.util.PMVMatrix;

public class SimpleExampleIco implements GLEventListener {
  Object3D obj;
  PMVMatrix mats;
  Shader shader;
  int uniformMat;
  int uniformLight;
  float t = 0;
  static final int SCREENH = 320;
  static final int SCREENW = 320;

  public SimpleExampleIco() {
    obj = new Icosahedron();
    // obj = new Cylinder(16,.7f,.5f,true);
    // obj = new GridPlane(12,5,8f,3f);
    // obj = new BezierPatch();
    mats = new PMVMatrix();
    shader = new Shader("titech/CSC.T463.ComputerGraphics/h6/resource/spot.vert",
        "titech/CSC.T463.ComputerGraphics/h6/resource/spot.frag");
    /*
    addKeyListener(new simpleExampleKeyListener());
    addMouseMotionListener(new simpleExampleMouseMotionListener());
    addMouseListener(new simpleExampleMouseListener());
    */
  }

  public void init(GLAutoDrawable drawable) {
    drawable.setGL(new DebugGL2(drawable.getGL().getGL2()));
    final GL2 gl = drawable.getGL().getGL2();
    // drawable.getGL().getGL2();
    gl.glViewport(0, 0, SCREENW, SCREENH);

    // Clear color buffer with black
    gl.glClearColor(1.0f, 1.0f, 0.5f, 1.0f);
    gl.glClearDepth(1.0f);
    gl.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT);
    gl.glEnable(GL2.GL_DEPTH_TEST);
    gl.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1);
    gl.glFrontFace(GL.GL_CCW);
    gl.glEnable(GL.GL_CULL_FACE);
    gl.glCullFace(GL.GL_BACK);

    shader.init(gl);
    int programName = shader.getID();
    gl.glBindAttribLocation(programName, Object3D.VERTEXPOSITION, "inposition");
    gl.glBindAttribLocation(programName, Object3D.VERTEXCOLOR, "incolor");
    gl.glBindAttribLocation(programName, Object3D.VERTEXNORMAL, "innormal");
    gl.glBindAttribLocation(programName, Object3D.VERTEXTEXCOORD0, "intexcoord0");
    shader.link(gl);
    uniformMat = gl.glGetUniformLocation(programName, "mat");
    uniformLight = gl.glGetUniformLocation(programName, "lightpos");
    gl.glUseProgram(programName);
    gl.glUniform3f(uniformLight, 0f, 0f, 10.0f);
    obj.init(gl, mats, programName);
    gl.glUseProgram(0);
  }

  public void display(GLAutoDrawable drawable) {
    final GL2 gl = drawable.getGL().getGL2();
    gl.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT);
    mats.glMatrixMode(GL2.GL_MODELVIEW);
    mats.glLoadIdentity();
    mats.glTranslatef(0f, 0f, -2.0f); // set the origin to (0,0,-2)
    if (t < 360) {
      t = t + 0.5f;
    } else {
      t = 0f;
    }
    mats.glRotatef(t, 0f, 1f, 0f);
    mats.glMatrixMode(GL2.GL_PROJECTION);
    mats.glLoadIdentity();
    mats.glFrustumf(-1f, 1f, -1f, 1f, 1f, 100f);
    mats.update();
    gl.glUseProgram(shader.getID());
    gl.glUniformMatrix4fv(uniformMat, 4, false, mats.glGetPMvMvitMatrixf());

    obj.display(gl, mats);
    // gl.glFlush();
    gl.glUseProgram(0);
  }

  public void reshape(GLAutoDrawable drawable, int x, int y, int w, int h) {}

  public void dispose(GLAutoDrawable drawable) {}

  public KeyListener createKeyListener() {
    return new simpleExampleKeyListener();
  }

  public MouseListener createMouseListener() {
    return new simpleExampleMouseListener();
  }

  public static void main(String[] args) {
    SimpleExampleIco t = new SimpleExampleIco();
    new SimpleExampleBase("SimpleExampleIco", t, SCREENW, SCREENH)
        .addKeyListener(t.createKeyListener())
        .addMouseListener(t.createMouseListener())
        .start();
  }

  class simpleExampleKeyListener implements KeyListener {
    public void keyPressed(KeyEvent e) {
      int keycode = e.getKeyCode();
      System.out.print(keycode);
      switch (keycode) {
        case com.jogamp.newt.event.KeyEvent.VK_LEFT:
          System.out.println("left key");
          break;
        case com.jogamp.newt.event.KeyEvent.VK_RIGHT:
          System.out.println("right key");
          break;
        case com.jogamp.newt.event.KeyEvent.VK_UP:
          System.out.println("up key");
          break;
        case com.jogamp.newt.event.KeyEvent.VK_DOWN:
          System.out.println("down key");
          break;
      }
    }
    public void keyReleased(KeyEvent e) {}
    public void keyTyped(KeyEvent e) {}
  }

  class simpleExampleMouseListener implements MouseListener {
    public void mouseDragged(MouseEvent e) {
      System.out.println("dragged:" + e.getX() + " " + e.getY());
    }
    public void mouseMoved(MouseEvent e) {
      System.out.println("moved:" + e.getX() + " " + e.getY());
    }
    public void mouseWheelMoved(MouseEvent e) {}
    public void mouseClicked(MouseEvent e) {
      System.out.println("clicked:" + e.getX() + " " + e.getY());
    }
    public void mouseEntered(MouseEvent e) {
      System.out.println("entered:");
    }
    public void mouseExited(MouseEvent e) {
      System.out.println("exited:");
    }
    public void mousePressed(MouseEvent e) {
      System.out.println("pressed:" + e.getX() + " " + e.getY());
    }
    public void mouseReleased(MouseEvent e) {
      System.out.println("released:" + e.getX() + " " + e.getY());
    }
  }
}
