import java.io.*;
import javax.imageio.*;
import java.nio.*;
import java.awt.image.*;

public class ImageLoader implements TextureImage{
  final private ByteBuffer data;
  final public int width, height;
  public ByteBuffer tmpdata=null;

  public ImageLoader(String fname){
    File imagefile=null;
    BufferedImage bimage = null;
    ByteBuffer tmp=null;
    int tmpw=0, tmph=0;
    try {
      imagefile = new File(fname);
      bimage = ImageIO.read(imagefile);
      tmpw = bimage.getWidth();
      tmph = bimage.getHeight();
      tmp = ByteBuffer.allocate(tmpw*tmph*4);
      tmpdata = ByteBuffer.allocate(tmpw*tmph*4);
      for(int y=0;y<tmph;y++){
        for(int x=0;x<tmpw;x++){
          int c = bimage.getRGB(x,y);
          tmp.putInt(((c&0xff0000)>>8)|((c&0xff00)<<8)|((c&0xff)<<24)|255);
	}
      }
      tmp.rewind();
    }catch(IOException e){
      System.err.println(e);
      System.exit(1);
    }finally{
      data = tmp;
      width = tmpw;
      height = tmph;
    }
  }

  public int getWidth(){
    return width;
  }

  public int getHeight(){
    return height;
  }

  public ByteBuffer getByteBuffer(){
    data.rewind();
    return data;
  }

  public ByteBuffer getByteBufferOfLevel(int level){
    TextureImageUtil.subsampling(data,tmpdata, width, height, level);
    return tmpdata;
  }

  public void save(){
    BufferedImage img = new BufferedImage(width, height,
                              //BufferedImage.TYPE_3BYTE_BGR);
                              BufferedImage.TYPE_4BYTE_ABGR);
    int r,g,b,a;
    ByteBuffer buff = data;
    buff.rewind();
    for(int y=0;y<height;y++){
      for(int x=0;x<width;x++){
	/*  in case of BGRA format for Texture */
	b = (int)buff.get();
	g = (int)buff.get();
	r = (int)buff.get();
	a = (int)buff.get();
	img.setRGB(x,y,(((((a<<8)|b)<<8)|g)<<8)|r);
      }
    }
    try{
      File file = new File("out.png");
      ImageIO.write((RenderedImage)img,"png",file);
    }catch(IOException ex){
    }
  }
}
