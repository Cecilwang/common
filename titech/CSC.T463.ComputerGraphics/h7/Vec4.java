
public class Vec4{
  public float[] data;

  public Vec4(){
    data = new float[4];
  }

  public Vec4(float a, float b, float c, float d){
    data = new float[4];
    data[0] = a;
    data[1] = b;
    data[2] = c;
    data[3] = d;
  }

  public Vec4(float[] in){
    data = new float[4];
    data[0] = in[0]; 
    data[1] = in[1]; 
    data[2] = in[2]; 
    data[3] = in[3]; 
  }

  static float dot(Vec4 a, Vec4 b){
    return a.data[0]*b.data[0]+a.data[1]*b.data[1]
      +a.data[2]*b.data[2]+a.data[3]*b.data[3];
  }

}