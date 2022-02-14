import javax.imageio.ImageIO;
import java.nio.*;
import java.io.*;
import java.awt.image.BufferedImage;
import com.jogamp.opengl.*;
import com.jogamp.newt.opengl.*;
import com.jogamp.opengl.util.FPSAnimator;
import com.jogamp.opengl.util.Animator;
import com.jogamp.newt.event.MouseListener;
import com.jogamp.newt.event.KeyListener;
import com.jogamp.newt.event.WindowEvent;
import com.jogamp.newt.event.WindowAdapter;
import com.jogamp.opengl.GLDebugMessage;
import com.jogamp.opengl.GLDebugListener;

/**
 * test launch class for Scene
 */

public class SimpleExampleBase{
  GLWindow window;
  FPSAnimator animator;
  GLDebugListener debuglistener;
  ByteBuffer buffer;

  final public static int WIDTH =300;
  final public static int HEIGHT =300;
  final public int width;
  final public int height;
  final public DebugMessageOut onoff;
  final private GLEventListener dsp;

  public SimpleExampleBase(String name, GLEventListener dsp, DebugMessageOut onoff){
    this(name, dsp, WIDTH, HEIGHT, onoff);
  }

  public SimpleExampleBase(String name, GLEventListener dsp){
    this(name, dsp, DebugMessageOut.OFF);
  }

  public SimpleExampleBase(String name, GLEventListener dsp, int width, int height){
    this(name, dsp, width, height, DebugMessageOut.OFF, null);
  }

  public SimpleExampleBase(String name, GLEventListener dsp, int width, int height,
                       DebugMessageOut onoff){
    this(name, dsp, width, height, onoff, null);
  }

  public SimpleExampleBase(String name, GLEventListener dsp, int width, int height,
                              DebugMessageOut onoff,
                              DebugMessageListener pdebuglistener){
    this.dsp = dsp;
    this.width = width;
    this.height = height;
    buffer = ByteBuffer.allocate(width * height * 4);
    if(pdebuglistener == null){
      this.debuglistener =
        new DebugMessageListener(System.err,
                                 DebugMessageListener.MessageLevel.ERROR);
    }else{
      this.debuglistener = pdebuglistener;
    }
    System.out.println("****\n default profile: "+ GLProfile.getDefault()+"\n****" );
    GLProfile profile;
    profile = GLProfile.get(GLProfile.GL2);
    GLCapabilities capabilities = new GLCapabilities(profile);
    capabilities.setDoubleBuffered(true);
    capabilities.setNumSamples(8); // enable anti aliasing - just as a example
    capabilities.setSampleBuffers(true);
    window = GLWindow.create(capabilities);
    window.setSize(width,height);
    animator = new FPSAnimator(window, 60, true);
    window.addWindowListener( new WindowAdapter(){
        public void windowDestroyNotify(WindowEvent e){
          animator.stop();
          System.exit(0);
        }
        public void windowDestroyed(WindowEvent e){
        }
      });
    window.addGLEventListener(dsp);
    window.setTitle(name);
    this.onoff = onoff;
    if(onoff == DebugMessageOut.ON){
      window.setContextCreationFlags(GLContext.CTX_OPTION_DEBUG);
    }
    if(onoff == DebugMessageOut.ON){
      window.invoke(false, new GLRunnable(){
        public boolean run(GLAutoDrawable drawable){
          drawable.getContext().addGLDebugListener(debuglistener);
          return true;
        }
        });
    }

  }

  public void start(){
    window.setVisible(true);
    window.setGL(new DebugGL2(window.getGL().getGL2()));
    //animator.setRunAsFastAsPossible(false);
    animator.setUpdateFPSFrames(100, System.out);
    animator.start();
  }

  public void draw(){
    window.setVisible(true);
    window.display();
  }

  /**
   * save current rendered shot as a png file.
   * This waits for finishing file save action
   * @param name file name
   */
  public void saveImage(String name){
    // true specifies block until execution of glRunnable is finished
    window.invoke(true, new GLRunnable(){
        public boolean run(GLAutoDrawable drawable){
          GL gl = drawable.getGL();
          gl.glFlush();
          gl.glFinish();
          buffer.rewind();
          gl.glReadPixels(0, 0, width, height, GL.GL_RGBA,
                          GL.GL_UNSIGNED_BYTE, buffer);
          BufferedImage image
            = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
          buffer.rewind();
          for(int y= height - 1; 0 <= y; y--){
            for(int x=0; x<width; x++){
              int r = buffer.get() & 0xff;
              int g = buffer.get() & 0xff;
              int b = buffer.get() & 0xff;
              int a = buffer.get() & 0xff;
              image.setRGB(x, y, (r<<16)|(g<<8)|(b)|(a<<24));
            }
          }
          try{
            ImageIO.write(image, "png", new File(name+".png"));
          }catch(IOException ex){
          }
          return true;
        }
        });
  }

  public SimpleExampleBase addMouseListener(MouseListener listener){
    window.addMouseListener(listener);
    return this;
  }

  public SimpleExampleBase addKeyListener(KeyListener listener){
    window.addKeyListener(listener);
    return this;
  }

  public enum DebugMessageOut {ON, OFF};
}
