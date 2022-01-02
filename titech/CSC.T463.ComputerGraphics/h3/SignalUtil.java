import java.awt.image.BufferedImage;
import java.awt.Color;
import java.awt.Graphics2D;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Scalar;
import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;


public class SignalUtil {

  private SignalUtil(){
  }

  static BufferedImage createGaussianImage(double sigmax, double sigmay){
    BufferedImage ret =
      new BufferedImage((int)(6 * sigmax + 1), (int)(6 * sigmay + 1),
                        BufferedImage.TYPE_INT_RGB);
    if(Math.abs(sigmax)<0.001 && Math.abs(sigmay)<0.001){
      ret.setRGB(0,0,0);
    }else{
      int h = ret.getHeight();
      int w = ret.getWidth();
      int halfh = h/2;
      int halfw = w/2;
      for(int y=0;y<h;y++){
        for(int x=0;x<w;x++){
          int val =(int)
            (Math.exp(-(x-halfw)*(x-halfw)/(2*sigmax*sigmax))*
             Math.exp(-(y - halfh) * (y - halfh) / (2 * sigmay * sigmay)) * 255);
          //  / (2 * Math.PI * sigmax * sigmay));
          ret.setRGB(x,y,((val&0xff)<<16)+((val&0xff)<<8)+ (val&0xff));
        }
      }
    }
    return ret;
  }

  static BufferedImage createGaussianMaskImage(int width, int height,
                                               double sigmax, double sigmay){
    BufferedImage ret =
      new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
    if(Math.abs(sigmax)<0.001 && Math.abs(sigmay)<0.001){
      Graphics2D gc = ret.createGraphics();
      gc.setColor(Color.WHITE);
      gc.fillRect(0, 0, width, height);
    }else{
      int w = width;
      int h = height;
      int halfw = w/2;
      int halfh = h/2;
      for(int y=0;y<h;y++){
        for(int x=0;x<w;x++){
          int val =(int)((1.0-Math.exp(-(x-halfw)*(x-halfw)/(2*sigmax*sigmax))*
              Math.exp(-(y - halfh) * (y - halfh) / (2 * sigmay * sigmay))) * 255);
          //  / (2 * Math.PI * sigmax * sigmay));
          ret.setRGB(x,y,((val&0xff)<<16)+((val&0xff)<<8)+ (val&0xff));
        }
      }
    }
    return ret;
  }

  static public void FourierTransform(TwoDimData input, TwoDimData output){
    // make MatVector planes which has two single-channel Mat, 'padded' and 'zeros'
    Mat matcomp = new Mat();
    merge(input.current, matcomp); // merge two Mats of MatVector into one 2-channel Mat 'complexI'
    Mat result = new Mat();
    dft(matcomp, result); // discrete fourier transform
    swapQuadrant(result);
    MatVector tmpvec = new MatVector(2);
    split(result, tmpvec);
    output.setDefault(tmpvec.get(0), tmpvec.get(1));
  }

  static public void InverseFourierTransform(TwoDimData input, TwoDimData output){
    // make MatVector planes which has two single-channel Mat, 'padded' and 'zeros'
    Mat matcomp = new Mat();
    merge(input.current, matcomp); // merge two Mats of MatVector into one 2-channel Mat 'complexI'
    swapQuadrant(matcomp);
    Mat result = new Mat();
    // discrete inverse fourier transform
    idft(matcomp, result, DFT_SCALE, 0);
    MatVector tmpvec = new MatVector(2);
    split(result, tmpvec);
    output.setDefault(tmpvec.get(0), tmpvec.get(1));
  }

  static private void swapQuadrant(Mat mat){
    int cx = mat.cols() / 2; //center x
    int cy = mat.rows() / 2; //center y
     // Create a ROI(Region Of Interest) per quadrant. They are not copies, but pointers.
    Mat q1 = new Mat(mat, new Rect(cx, 0, cx, cy)); // Top-Right
    Mat q2 = new Mat(mat, new Rect(0, 0, cx, cy)); // Top-Left
    Mat q3 = new Mat(mat, new Rect(0, cy, cx, cy)); // Bottom-Left
    Mat q4 = new Mat(mat, new Rect(cx, cy, cx, cy)); // Bottom-Right
    Mat tmp = new Mat();
    // swap quadrants (Top-Left with Bottom-Right)
    q2.copyTo(tmp);
    q4.copyTo(q2);
    tmp.copyTo(q4);
    // swap quadrant (Top-Right with Bottom-Left)
    q1.copyTo(tmp);
    q3.copyTo(q1);
    tmp.copyTo(q3);
  }
}
