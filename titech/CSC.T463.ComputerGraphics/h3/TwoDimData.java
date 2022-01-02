import java.io.File;
import java.awt.image.BufferedImage;
import org.bytedeco.opencv.opencv_core.*;
//import org.bytedeco.javacpp.indexer.Indexer;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class TwoDimData {
  int width;
  int height;
  final private MatVector original;
  final MatVector current;

  public TwoDimData(int width, int height){
    this.width = width;
    this.height = height;
    original = new MatVector(2);
    current = new MatVector(2);
  }

  public void normalizeCurrentL1() {
    double sum0 = sumElems(current.get(0)).get(0);
    double sum1 = sumElems(current.get(1)).get(0);
    if(sum0!=0.0){
      current.put(0, multiply(current.get(0), 1.0 / sum0).asMat()); // MatExpr
    }
    if(sum1!=0.0){
      current.put(1, multiply(current.get(1), 1.0 / sum1).asMat()); // MatExpr
    }
  }

  public void normalizeCurrentL2() {
    Mat powmat = new Mat(new int[]{width, height}, CV_32FC1);
    multiply(current.get(0),current.get(0), powmat);
    double sum0 = Math.sqrt(sumElems(powmat).get(0));
    if(sum0!=0.0){
      current.put(0, multiply(current.get(0), 1.0 / sum0).asMat()); // MatExpr
    }
    multiply(current.get(1),current.get(1), powmat);
    double sum1 = Math.sqrt(sumElems(powmat).get(0));
    if(sum1!=0.0){
      current.put(1, multiply(current.get(1), 1.0 / sum1).asMat()); // MatExpr
    }
  }

  public void setDefault(Mat real, Mat imag) {
    width = real.rows();
    height = real.cols();
    real.copyTo(original.get(0));
    imag.copyTo(original.get(1));
    current.put(0, original.get(0).clone());
    current.put(1, original.get(1).clone());
  }

  public void setDefault(Mat data){
    width = data.rows();
    height = data.cols();
    data.copyTo(original.get(0));
    Mat.zeros(data.size(),CV_32FC1).asMat().copyTo(original.get(1));
    current.put(0, original.get(0).clone());
    current.put(1, original.get(1).clone());
  }

  public void setDefault(TwoDimData input){
    width = input.width;
    height = input.height;
    input.current.get(0).copyTo(original.get(0));
    input.current.get(1).copyTo(original.get(1));
    reset();
  }

  public void setDefault(BufferedImage greenimage) {
    width = greenimage.getWidth();
    height = greenimage.getHeight();
    original.get(0).create(width, height, CV_32FC1);
    Mat.zeros(width, height, CV_32FC1).asMat().copyTo(original.get(1));
    FloatIndexer indexer = original.get(0).createIndexer();
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        indexer.put(y, x, ((greenimage.getRGB(x, y) >> 8) & 0xff)/255.0f);
      }
    }
    reset();
  }

  public void setDefault(BufferedImage greenimage0,
                         BufferedImage greenimage1) {
    width = greenimage0.getWidth();
    height = greenimage0.getHeight();
    original.get(0).create(width, height, CV_32FC1);
    original.get(1).create(width, height, CV_32FC1);
    FloatIndexer indexer0 = original.get(0).createIndexer();
    FloatIndexer indexer1 = original.get(1).createIndexer();
    for  (int  y  = 0 ;  y < height;  y++) {
      for  (int  x  = 0 ;  x < width;  x++) {
        indexer0.put(y, x, ((greenimage0.getRGB(x,  y )  >>  8) & 0xff)/255.0f);
        indexer1.put(y, x, ((greenimage1.getRGB(x,  y )  >>  8) & 0xff)/255.0f);
      }
    }
    reset();
  }

  public void setResizeAndDefault(File file) {
    Mat tmp = imread(file.getAbsolutePath(), IMREAD_GRAYSCALE);
    width = tmp.cols();
    height = tmp.rows();
    Mat tmp2 = new Mat(tmp, new Rect(0, 0, width, height));
    tmp2.convertTo(original.get(0), CV_32FC1);//transform to float data
    tmp2 = dividePut(original.get(0), 255.0); //max value = 1
    tmp2.copyTo(original.get(0));
    Mat.zeros(tmp2.size(), CV_32FC1).asMat().copyTo(original.get(1));
    reset();
  }

  public void setDefault(File file) {
    Mat tmp = imread(file.getAbsolutePath(), IMREAD_GRAYSCALE);
    Mat tmp2 = new Mat(tmp, new Rect(0, 0, width, height));
    tmp2.convertTo(original.get(0), CV_32FC1);//transform to float data
    tmp2 = dividePut(original.get(0), 255.0); //max value = 1
    tmp2.copyTo(original.get(0));
    Mat.zeros(tmp2.size(), CV_32FC1).asMat().copyTo(original.get(1));
    reset();
  }

  public void setDefault(File file0, File file1) {
    Mat tmp = imread(file0.getAbsolutePath(),IMREAD_GRAYSCALE);
    Mat tmp2 = new Mat(tmp, new Rect(0,0,width,height));
    tmp2.convertTo(original.get(0), CV_32FC1);//transform to float data
    tmp2 = dividePut(original.get(0), 255.0); //max value = 1
    tmp2.copyTo(original.get(0));
    tmp = imread(file1.getAbsolutePath(),IMREAD_GRAYSCALE);
    tmp2 = new Mat(tmp, new Rect(0,0,width,height));
    tmp2.convertTo(original.get(1), CV_32FC1);//transform to float data
    tmp2 = dividePut(original.get(1), 255.0); //max value = 1
    tmp2.copyTo(original.get(1));
    reset();
  }

  public void reset() {
    current.put(0, original.get(0).clone());
    current.put(1, original.get(1).clone());
  }

  public void convolutionWith(TwoDimData kernel) {
    Mat tmp = current.get(0).clone();
    filter2D(tmp, current.get(0), -1, kernel.current.get(0));
    tmp = current.get(1).clone();
    filter2D(tmp, current.get(1), -1, kernel.current.get(1));
  }

  public TwoDimData convolutionOut(TwoDimData kernel) {
    TwoDimData ret = new TwoDimData(width, height);
    filter2D(current.get(0), ret.original.get(0), -1, kernel.current.get(0));
    filter2D(current.get(1), ret.original.get(1), -1, kernel.current.get(1));
    ret.reset();
    return ret;
  }

  public void multiplyWith(double scalar) {
    current.put(0, multiply(current.get(0), scalar).asMat());
    current.put(1, multiply(current.get(1), scalar).asMat());
  }

  public TwoDimData multiplyOut(double scalar) {
    TwoDimData ret = new TwoDimData(width, height);
    ret.original.put(0, multiply(current.get(0), scalar).asMat());
    ret.original.put(1, multiply(current.get(1), scalar).asMat());
    ret.reset();
    return ret;
  }

  public void multiplyWith(TwoDimData another) {
    multiply(current.get(0), another.current.get(0), current.get(0));//multiply each elements
    multiply(current.get(1), another.current.get(1), current.get(1));//multiply each elements
    //current.put(0, multiply(current.get(0), another.current.get(0)).asMat());//bad
    //current.put(1, multiply(current.get(1), another.current.get(1)).asMat());//bad
  }

  public TwoDimData multiplyOut(TwoDimData another) {
    TwoDimData ret = new TwoDimData(width, height);
    multiply(current.get(0), another.current.get(0), ret.original.get(0));//multiply each elements
    multiply(current.get(1), another.current.get(1), ret.original.get(1));//multiply each elements
    ret.reset();
    return ret;
  }

  public void subtractWith(TwoDimData another) {
    current.put(0, subtract(current.get(0), another.current.get(0)).asMat());
    current.put(1, subtract(current.get(1), another.current.get(1)).asMat());
  }

  public void subtractWith(double rl, double ig) {
    current.put(0, subtract(current.get(0), new Scalar(rl)).asMat());
    current.put(1, subtract(current.get(1), new Scalar(ig)).asMat());
  }

  public TwoDimData subtractOut(TwoDimData another) {
    TwoDimData ret = new TwoDimData(width, height);
    ret.original.put(0,
        subtract(current.get(0), another.current.get(0)).asMat());
    ret.original.put(1,
        subtract(current.get(1), another.current.get(1)).asMat());
    ret.reset();
    return ret;
  }

  public TwoDimData subtractOut(double rl, double ig) {
    TwoDimData ret = new TwoDimData(width, height);
    ret.original.put(0,
        subtract(current.get(0), new Scalar(rl)).asMat());
    ret.original.put(1,
        subtract(current.get(1), new Scalar(ig)).asMat());
    ret.reset();
    return ret;
  }

  public void addWith(TwoDimData another) {
    current.put(0, add(current.get(0), another.current.get(0)).asMat());
    current.put(1, add(current.get(1), another.current.get(1)).asMat());
  }

  public void addWith(double rl, double ig) {
    current.put(0, add(current.get(0), new Scalar(rl)).asMat());
    current.put(1, add(current.get(1), new Scalar(ig)).asMat());
  }

  public TwoDimData addOut(TwoDimData another) {
    TwoDimData ret = new TwoDimData(width, height);
    ret.original.put(0, add(current.get(0), another.current.get(0)).asMat());
    ret.original.put(1, add(current.get(1), another.current.get(1)).asMat());
    ret.reset();
    return ret;
  }

  public TwoDimData addOut(double rl, double ig) {
    TwoDimData ret = new TwoDimData(width, height);
    ret.original.put(0, add(current.get(0), new Scalar(rl)).asMat());
    ret.original.put(1, add(current.get(1), new Scalar(ig)).asMat());
    ret.reset();
    return ret;
  }

  public BufferedImage exportBufferedImage() {
     BufferedImage ret =  new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB);
     FloatIndexer indexer = current.get(0).createIndexer();
    for(int y=0; y<height; y++){
      for(int x=0; x<width; x++){
        int val = (int)(indexer.get(y, x)*255.0f);
        ret.setRGB(x, y, ((val&0xff)<<16)+((val&0xff)<<8)+ (val&0xff));
      }
    }
    return ret;
  }

  public BufferedImage exportBiasedBufferedImage() {
     BufferedImage ret =  new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB);
     FloatIndexer indexer = current.get(0).createIndexer();
    for(int y=0; y<height; y++){
      for(int x=0; x<width; x++){
        int val = (int)(indexer.get(y, x)*255.0f/2.0f)+128;
        ret.setRGB(x, y, ((val&0xff)<<16)+((val&0xff)<<8)+ (val&0xff));
      }
    }
    return ret;
  }

  public BufferedImage exportMagnitudeLoggedBufferedImage() {
    BufferedImage ret =  new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB);
    FloatIndexer indexer0 = current.get(0).createIndexer();
    FloatIndexer indexer1 = current.get(0).createIndexer();
    double maxvalue=0;
    for(int y=0; y<height; y++){
      for(int x=0; x<width; x++){
        double val0 = indexer0.get(y, x);
        double val1 = indexer1.get(y, x);
        double val = Math.log1p(Math.sqrt(val0*val0+val1*val1)); //log_e(x+1)
        if(maxvalue< val){
          maxvalue = val;
        }
      }
    }
    for(int y=0; y<height; y++){
      for(int x=0; x<width; x++){
        double val0 = indexer0.get(y, x);
        double val1 = indexer1.get(y, x);
        int val = (int)(Math.log1p(Math.sqrt(val0*val0+val1*val1))*255/maxvalue); //log_e(x+1)
        ret.setRGB(x, y, ((val&0xff)<<16)+((val&0xff)<<8)+ (val&0xff));
      }
    }
    return ret;
  }

  public void printCurrentReal(){
    FloatIndexer idx = current.get(0).createIndexer();
    System.out.println("# kernel matrix");
    for(int i=0;i< current.get(0).rows();i++){
      for (int ii = 0; ii < current.get(0).cols(); ii++) {
        System.out.print(idx.get(i, ii) + ",");
      }
      System.out.println();
    }
  }
}
