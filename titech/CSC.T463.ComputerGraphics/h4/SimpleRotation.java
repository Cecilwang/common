
import com.jogamp.newt.event.KeyEvent;
import com.jogamp.newt.event.KeyListener;
import com.jogamp.newt.event.MouseEvent;
import com.jogamp.newt.event.MouseListener;
import com.jogamp.opengl.*;
import com.jogamp.opengl.util.PMVMatrix;

public class SimpleRotation implements GLEventListener {
  final Object3D obj;
  final PMVMatrix mats;
  final Shader shader;
  int uniformMat;
  int uniformLight;
  float t = 0;
  static final int SCREENH = 320;
  static final int SCREENW = 320;
  static final float SOLAR_ROTATION_PERIOD = 24.47f;

  static final float EARTH_ORBITAL_PERIOD = 365f;
  static final float EARTH_ORBITAL_INCLINATION = 7f;
  static final float REAL_EARTH_ROTATION_PERIOD = 1f;
  static final float EARTH_ROTATION_PERIOD = REAL_EARTH_ROTATION_PERIOD * 5f;
  static final float REAL_SIZE_RATIO_EARTH_SUN = 0.009174311927f;
  static final float SIZE_RATIO_EARTH_SUN = REAL_SIZE_RATIO_EARTH_SUN * 30f;
  static final float EARTH_OBLIQUITY = 23.439f;
  static final float REAL_EARTH_ORBITAL_A = 152f;
  static final float REAL_EARTH_ORBITAL_B = 147f;
  static final float EARTH_ORBITAL_A = REAL_EARTH_ORBITAL_A / 30f;
  static final float EARTH_ORBITAL_B =
      REAL_EARTH_ORBITAL_B / REAL_EARTH_ORBITAL_A * EARTH_ORBITAL_A;

  static final float MOON_ORBITAL_PERIOD = 27f;
  static final float MOON_ROTATION_PERIOD = 27f;
  static final float SIZE_RATIO_MOON_SUN = SIZE_RATIO_EARTH_SUN * 0.27f;
  static final float REAL_MOON_ORBITAL_A = 0.405f;
  static final float REAL_MOON_ORBITAL_B = 0.364f;
  static final float MOON_ORBITAL_A = 2 * REAL_MOON_ORBITAL_A;
  static final float MOON_ORBITAL_B = REAL_MOON_ORBITAL_B / REAL_MOON_ORBITAL_A * MOON_ORBITAL_A;

  static int PrintI = 0;

  public SimpleRotation() {
    obj = new Cube();
    mats = new PMVMatrix();
    shader = new Shader("titech/CSC.T463.ComputerGraphics/h4/resource/simple.vert",
        "titech/CSC.T463.ComputerGraphics/h4/resource/simple.frag");
  }

  public void init(GLAutoDrawable drawable) {
    final GL2GL3 gl = drawable.getGL().getGL2GL3();
    if (gl.isGL4()) {
      drawable.setGL(new DebugGL4(drawable.getGL().getGL4()));
    } else if (gl.isGL3()) {
      drawable.setGL(new DebugGL3(drawable.getGL().getGL3()));
    } else if (gl.isGL2()) {
      drawable.setGL(new DebugGL2(drawable.getGL().getGL2()));
    }
    // drawable.getGL().getGL2();
    gl.glViewport(0, 0, SCREENW, SCREENH);

    // Clear color buffer with black
    gl.glClearColor(1.0f, 0.5f, 1.0f, 1.0f);
    gl.glClearDepth(1.0f);
    gl.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT);
    gl.glEnable(GL2.GL_DEPTH_TEST);
    gl.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1);
    gl.glFrontFace(GL.GL_CCW);
    gl.glEnable(GL.GL_CULL_FACE);
    gl.glCullFace(GL.GL_BACK);

    gl.glCreateShader(GL2GL3.GL_VERTEX_SHADER);
    shader.init(gl);
    int programName = shader.getID();
    gl.glBindAttribLocation(programName, Object3D.VERTEXPOSITION, "inposition");
    gl.glBindAttribLocation(programName, Object3D.VERTEXCOLOR, "incolor");
    gl.glBindAttribLocation(programName, Object3D.VERTEXNORMAL, "innormal");
    gl.glBindAttribLocation(programName, Object3D.VERTEXTEXCOORD0, "intexcoord0");
    shader.link(gl);
    uniformMat = gl.glGetUniformLocation(programName, "mat");
    uniformLight = gl.glGetUniformLocation(programName, "lightdir");
    gl.glUseProgram(programName);
    gl.glUniform3f(uniformLight, 0f, 10f, -10f);
    obj.init(gl, mats, programName);
    gl.glUseProgram(0);
  }

  public void print(PMVMatrix mat) {
    if (PrintI++ < 2) {
      System.out.println(mat.toString());
    }
  }

  public void prepareMat() {
    mats.glLoadIdentity(); /* set 4x4 identity matrix, I*/
    // set the origin to (0,0,-18)
    mats.glTranslatef(0f, 0f, -18.0f); /* multiply translation matrix, I*T */
  }

  public void displayOBJ(GL2GL3 gl) {
    mats.update();
    gl.glUniformMatrix4fv(uniformMat, 4, false, mats.glGetPMvMvitMatrixf());
    obj.display(gl, shader.getID());
  }

  public void displaySun(float t, GL2GL3 gl) {
    prepareMat();

    // Solar rotation
    float solar_t = (t * EARTH_ORBITAL_PERIOD / SOLAR_ROTATION_PERIOD) % 360f;
    mats.glRotatef(solar_t, 0f, 1f, 0f);

    displayOBJ(gl);
  }

  public void displayEarth(float t, GL2GL3 gl) {
    prepareMat();

    // The orbital rotation of Earth
    float x = EARTH_ORBITAL_A * (float) Math.cos(-t / 180f * Math.PI);
    float z = EARTH_ORBITAL_B * (float) Math.sin(-t / 180f * Math.PI);
    float y = (float) (Math.tan(EARTH_ORBITAL_INCLINATION / 180f * Math.PI) * x);
    mats.glTranslatef(x, y, z);

    // Earth rotation
    float earth_t = (t * EARTH_ORBITAL_PERIOD / EARTH_ROTATION_PERIOD) % 360f;
    mats.glRotatef(earth_t, (float) (Math.tan(EARTH_OBLIQUITY / 180f * Math.PI)), 1f, 0f);

    // Earth size
    mats.glScalef(SIZE_RATIO_EARTH_SUN, SIZE_RATIO_EARTH_SUN, SIZE_RATIO_EARTH_SUN);

    displayOBJ(gl);
  }

  public void displayMoon(float t, GL2GL3 gl) {
    prepareMat();

    // The orbital rotation of Earth
    float x = EARTH_ORBITAL_A * (float) Math.cos(-t / 180f * Math.PI);
    float z = EARTH_ORBITAL_B * (float) Math.sin(-t / 180f * Math.PI);
    float y = (float) (Math.tan(EARTH_ORBITAL_INCLINATION / 180f * Math.PI) * x);
    // The orbital rotation of Moon
    float moon_orbit_t = (t * EARTH_ORBITAL_PERIOD / MOON_ORBITAL_PERIOD) % 360f;
    x += MOON_ORBITAL_A * (float) Math.cos(-moon_orbit_t / 180f * Math.PI);
    z += MOON_ORBITAL_B * (float) Math.sin(-moon_orbit_t / 180f * Math.PI);
    mats.glTranslatef(x, y, z);

    // Earth rotation
    float moon_rotation_t = (t * EARTH_ORBITAL_PERIOD / MOON_ROTATION_PERIOD) % 360f;
    mats.glRotatef(moon_rotation_t, 0, 1f, 0f);

    // Moon size
    mats.glScalef(SIZE_RATIO_MOON_SUN, SIZE_RATIO_MOON_SUN, SIZE_RATIO_MOON_SUN);

    displayOBJ(gl);
  }

  public void display(GLAutoDrawable drawable) {
    final GL2GL3 gl = drawable.getGL().getGL2GL3();
    gl.glUseProgram(shader.getID());
    gl.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT);
    mats.glMatrixMode(GL2.GL_PROJECTION);
    mats.glLoadIdentity();
    mats.glFrustumf(-0.5f, 0.5f, -0.5f, 0.5f, 1f, 100f);
    mats.glMatrixMode(GL2.GL_MODELVIEW);

    t = (t + 0.3f) % 360f;
    displaySun(t, gl);
    displayEarth(t, gl);
    displayMoon(t, gl);

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
    SimpleRotation t = new SimpleRotation();
    new SimpleExampleBase("SimpleRotation", t, SCREENW, SCREENH)
        .addKeyListener(t.createKeyListener())
        .addMouseListener(t.createMouseListener())
        .start();
  }

  class simpleExampleKeyListener implements KeyListener {
    public void keyPressed(KeyEvent e) {
      int keycode = e.getKeyCode();
      System.out.print(keycode);
      if (java.awt.event.KeyEvent.VK_LEFT == keycode) {
        System.out.print("a");
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
    public void mouseClicked(MouseEvent e) {}
    public void mouseEntered(MouseEvent e) {}
    public void mouseExited(MouseEvent e) {}
    public void mousePressed(MouseEvent e) {
      System.out.println("pressed:" + e.getX() + " " + e.getY());
    }
    public void mouseReleased(MouseEvent e) {}
  }
}
