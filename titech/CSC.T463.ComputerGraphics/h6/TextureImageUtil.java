import java.nio.*;

class TextureImageUtil{
  static void subsampling(ByteBuffer in, ByteBuffer out, int originalw, 
                          int originalh, int level){
    in.rewind();
    out.rewind();
    int step = 4*(1<<level);
    for(int y=0;y<originalh;y+=(1<<level)){
      for(int x=0;x<originalw;x+=(1<<level)){
        int base = (y*originalw+x)*4;
        out.put(in.get(base));
        out.put(in.get(base+1));
        out.put(in.get(base+2));
        out.put(in.get(base+3));
      }
    }
    out.rewind();
    in.rewind();
  }
}