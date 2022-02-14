import com.jogamp.newt.event.KeyEvent;
import com.jogamp.newt.event.KeyListener;
import com.jogamp.newt.event.MouseEvent;
import com.jogamp.newt.event.MouseListener;
import com.jogamp.opengl.*;
import com.jogamp.opengl.util.PMVMatrix;

public class SimpleExample6Objs implements GLEventListener {
  Object3D[] objs;
  PMVMatrix mats;
  float t = 0;
  Vec3 lightpos;
  Vec3 lightcolor;
  static final int SCREENH = 320;
  static final int SCREENW = 320;
  static String prefix = "titech/CSC.T463.ComputerGraphics/h7/";

  public SimpleExample6Objs() {
    objs = new Object3D[6];
    objs[0] = new Cylinder2(18, 0.3f, 0.8f, true, new Vec4(1.f, 1.f, 1.f, 1.0f),
        new ImageLoader(prefix + "resource/image/flatwhite.png"));
    objs[1] = new Cylinder2(4, 0.5f, 0.8f, false, new Vec4(0.199f, 0.819f, 0.572f, 1.0f),
        new ImageLoader(prefix + "resource/image/BrickNormalMap.png"));
    objs[2] = new Cylinder2(18, 0.3f, 0.8f, true, new Vec4(0.7f, 0.7f, 0.2f, 1.0f),
        new ImageLoader(prefix + "resource/image/NormalMap.png"));
    objs[3] = new Cylinder2(4, 0.3f, 0.8f, false, new Vec4(0.0f, 0.0f, 1.0f, 1.0f),
        new ImageLoader(prefix + "resource/image/NormalMap.png"));
    objs[4] = new Cylinder2(4, 0.3f, 0.8f, false, new Vec4(1.0f, 0.0f, 0.0f, 1.0f),
        new ImageLoader(prefix + "resource/image/NormalMap.png"));
    objs[5] = new Cylinder2(4, 0.3f, 0.8f, false, new Vec4(0.0f, 1.0f, 0.0f, 1.0f),
        new ImageLoader(prefix + "resource/image/wood.png"));

    // obj = new BezierPatch();
    mats = new PMVMatrix();
    /*
    addKeyListener(new simpleExampleKeyListener());
    addMouseMotionListener(new simpleExampleMouseMotionListener());
    addMouseListener(new simpleExampleMouseListener());
    */
  }

  public void init(GLAutoDrawable drawable) {
    drawable.setGL(new DebugGL2(drawable.getGL().getGL2()));
    final GL2 gl = drawable.getGL().getGL2();
    gl.glViewport(0, 0, SCREENW, SCREENH);

    // Clear color buffer with black
    gl.glClearColor(0.7f, 0.7f, 0.5f, 1.0f);
    gl.glClearDepth(1.0f);
    gl.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT);
    gl.glEnable(GL2.GL_DEPTH_TEST);
    gl.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1);
    gl.glFrontFace(GL.GL_CCW);
    gl.glEnable(GL.GL_CULL_FACE);
    gl.glCullFace(GL.GL_BACK);
    Shader chalkShader =
        new Shader(prefix + "resource/shader/chalk.vert", prefix + "resource/shader/chalk.frag");
    Shader brickShader =
        new Shader(prefix + "resource/shader/brick.vert", prefix + "resource/shader/brick.frag");
    Shader copperShader =
        new Shader(prefix + "resource/shader/copper.vert", prefix + "resource/shader/copper.frag");
    Shader plasticShader = new Shader(
        prefix + "resource/shader/plastic.vert", prefix + "resource/shader/plastic.frag");
    Shader woodShader =
        new Shader(prefix + "resource/shader/wood.vert", prefix + "resource/shader/wood.frag");
    Shader goldShader =
        new Shader(prefix + "resource/shader/gold.vert", prefix + "resource/shader/gold.frag");

    chalkShader.init(gl);
    brickShader.init(gl);
    copperShader.init(gl);
    plasticShader.init(gl);
    woodShader.init(gl);
    goldShader.init(gl);

    objs[0].init(gl, mats, chalkShader);
    objs[1].init(gl, mats, brickShader);
    objs[2].init(gl, mats, copperShader);
    objs[3].init(gl, mats, plasticShader);
    objs[4].init(gl, mats, goldShader);
    objs[5].init(gl, mats, woodShader);

    // objs.init(gl, mats, null);
    gl.glUseProgram(0);
    lightpos = new Vec3(0.0f, 0.0f, 30f);
    lightcolor = new Vec3(1f, 1f, 1f);
  }

  public void display(GLAutoDrawable drawable) {
    final GL2 gl = drawable.getGL().getGL2();
    if (t < 360) {
      t = t + 0.3f;
    } else {
      t = 0f;
    }
    gl.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT);
    mats.glMatrixMode(GL2.GL_PROJECTION);
    mats.glLoadIdentity();
    mats.glFrustumf(-0.5f, 0.5f, -0.5f, 0.5f, 1f, 100f);
    mats.glMatrixMode(GL2.GL_MODELVIEW);
    mats.glLoadIdentity();
    mats.glTranslatef(0f, 0f, -4.0f);
    for (int i = 0; i < 6; i++) {
      mats.glPushMatrix();
      mats.glTranslatef((i % 3) - 1f, 0.7f - (i / 3) * 1.4f, 0f);
      // mats.glTranslatef(0f,-0.4f+0.8f*(float)Math.sin((t+90)/180.0*Math.PI),0f);
      mats.glRotatef(t, 0.3f, 1f, 0f);
      mats.glRotatef(90, 1f, 0f, 0f);
      mats.glRotatef(45, 0f, 0f, 1f);
      mats.update();
      objs[i].display(gl, mats, lightpos, lightcolor);
      mats.glPopMatrix();
    }
    gl.glFlush();
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
    SimpleExample6Objs t = new SimpleExample6Objs();
    new SimpleExampleBase("SimpleExample6Objs", t, SCREENW, SCREENH)
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
