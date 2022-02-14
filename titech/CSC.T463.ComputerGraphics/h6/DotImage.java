import java.awt.image.*;
import java.awt.*;
import java.io.*;
import java.nio.*;
import javax.imageio.*;

class DotImage implements TextureImage{
  private BufferedImage image;
  private ByteBuffer buff;
  private ByteBuffer tmpbuff;
  private int height, width;

  public DotImage(int width, int height){
    this.height = height;
    this.width = width;
    Graphics2D graphics;
    image = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
    graphics = image.createGraphics();
    for(int y = 0; y<height; y++){
      for(int x = 0; x<width; x++){
        float v=(float)((1.0+Math.sin(x/8.0*Math.PI))*
                        (1.0+Math.sin(y/8.0*Math.PI))/4.0);
        graphics.setColor(new Color(v,v,v));
        graphics.drawLine(x,y,x,y);
      }
    }
    buff = ByteBuffer.allocate(height*width*4);
    tmpbuff = ByteBuffer.allocate(height*width*4);
    for(int y=0;y<height;y++){
      for(int x=0;x<width;x++){
        buff.putInt((image.getRGB(x,y)<<8)|255);
      }
    }
  }

  public void save(String fname){
    try{
      File file = new File(fname+".png");
      ImageIO.write((RenderedImage)image,"png",file);
    }catch(IOException ex){
    }
  }

  public int getWidth(){
    return width;
  }

  public int getHeight(){
    return height;
  }

  public ByteBuffer getByteBuffer(){
    buff.rewind();
    return buff;
  }

  public ByteBuffer getByteBufferOfLevel(int level){
    TextureImageUtil.subsampling(buff, tmpbuff, width, height, level);
    return tmpbuff;
  }

  public static void main(String[] args){
    DotImage di = new DotImage(512,512);
    di.save("dot");
  }
}

