import java.io.PrintStream;
import com.jogamp.opengl.GLDebugListener;
import com.jogamp.opengl.GLDebugMessage;
import static com.jogamp.opengl.GL2ES2.GL_DEBUG_SEVERITY_LOW;
import static com.jogamp.opengl.GL2ES2.GL_DEBUG_SEVERITY_MEDIUM;
import static com.jogamp.opengl.GL2ES2.GL_DEBUG_SEVERITY_HIGH;
import static com.jogamp.opengl.GL2ES2.GL_DEBUG_SEVERITY_NOTIFICATION;

public class DebugMessageListener implements GLDebugListener{
  private MessageLevel messagelevel;
  private PrintStream ostream;
  
  public DebugMessageListener(){
    messagelevel = MessageLevel.ERROR;
    ostream = System.err;
  }

  public DebugMessageListener(PrintStream stream){
    messagelevel = MessageLevel.ERROR;
    ostream = stream;
  }

  public DebugMessageListener(PrintStream stream, MessageLevel level){
    messagelevel = level;
    ostream = stream;
  }

  public DebugMessageListener(MessageLevel level){
    messagelevel = level;
    ostream = System.err;
  }
  
  public void messageSent(GLDebugMessage event){
    switch(event.getDbgSeverity()){
    case GL_DEBUG_SEVERITY_HIGH:
      showMessage(event);
      break;
    case GL_DEBUG_SEVERITY_MEDIUM:
      if(messagelevel.ordinal()<3){
        showMessageWithStack(event);
      }
      break;
    case GL_DEBUG_SEVERITY_LOW:
      if(messagelevel.ordinal()<2){
        showMessageWithStack(event);
      }
      break;
    case GL_DEBUG_SEVERITY_NOTIFICATION:
      if(messagelevel.ordinal()<1){
        showMessageWithStack(event);
      }      
    }
  }

  private void showMessage(GLDebugMessage event){
    ostream.println("******************************************");
    ostream.println(event);
  }
  private void showMessageWithStack(GLDebugMessage event){
    ostream.println("------------------------------------------");
    ostream.println(event);
    StackTraceElement[] elements = new Exception().getStackTrace();
    for(int i=4; i<elements.length; i++){
      ostream.println("  "+elements[i].toString());
    }
  }

  public enum MessageLevel {EVERYTHING, LOWIMPORTANT, WARNING, ERROR};
}
