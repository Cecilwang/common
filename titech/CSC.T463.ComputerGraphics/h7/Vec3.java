import Jama.Matrix;

public class Vec3{
  public float[] data;

  public Vec3(){
    data = new float[3];
  }

  public Vec3(float a, float b, float c){
    data = new float[3];
    data[0] = a;
    data[1] = b;
    data[2] = c;
  }

  public Vec3(float[] in){
    data = new float[3];
    data[0] = in[0]; 
    data[1] = in[1]; 
    data[2] = in[2]; 
  }

  public Vec3(Vec4 v){
    data = new float[3];
    data[0] = v.data[0]/v.data[3];
    data[1] = v.data[1]/v.data[3];
    data[2] = v.data[2]/v.data[3];
  }

  static float dot(Vec3 a, Vec3 b){
    return a.data[0]*b.data[0]+a.data[1]*b.data[1]+a.data[2]*b.data[2];
  }

  public void add(Vec3 a){
    data[0] +=a.data[0];
    data[1] +=a.data[1];
    data[2] +=a.data[2];
  }

  public void sub(Vec3 a){
    data[0] -=a.data[0];
    data[1] -=a.data[1];
    data[2] -=a.data[2];
  }

  public void mul(Vec3 a){
    data[0] *=a.data[0];
    data[1] *=a.data[1];
    data[2] *=a.data[2];
  }

  public void mul(float a){
    data[0] *=a;
    data[1] *=a;
    data[2] *=a;
  }

  public void normalize(){
    float t= dot(this,this);
    t = (float)Math.sqrt(t);
    mul(1.0f/t);
  }

  public static Vec3 normal(Vec3 a){
    float t= dot(a,a);
    t = (float)Math.sqrt(t);
    return times(a,1.0f/t);
  }

  public static Vec3 plus(Vec3 a, Vec3 b){
    Vec3 ret = new Vec3();
    ret.data[0] = a.data[0]+ b.data[0]; 
    ret.data[1] = a.data[1]+ b.data[1]; 
    ret.data[2] = a.data[2]+ b.data[2]; 
    return ret;
  }

  public static Vec3 minus(Vec3 a, Vec3 b){
    Vec3 ret = new Vec3();
    ret.data[0] = a.data[0]- b.data[0]; 
    ret.data[1] = a.data[1]- b.data[1]; 
    ret.data[2] = a.data[2]- b.data[2]; 
    return ret;
  }

  public static Vec3 times(Vec3 a, Vec3 b){
    Vec3 ret = new Vec3();
    ret.data[0] = a.data[0]* b.data[0]; 
    ret.data[1] = a.data[1]* b.data[1]; 
    ret.data[2] = a.data[2]* b.data[2]; 
    return ret;
  }

  public static Vec3 times(Vec3 a, float b){
    Vec3 ret = new Vec3();
    ret.data[0] = a.data[0]* b;
    ret.data[1] = a.data[1]* b;
    ret.data[2] = a.data[2]* b;
    return ret;
  }

  static Vec3 crossprod(Vec3 a, Vec3 b){
    Vec3 ret = new Vec3();
    ret.data[0] = a.data[1] * b.data[2] - b.data[1] * a.data[2];
    ret.data[1] = a.data[2] * b.data[0] - b.data[2] * a.data[0];
    ret.data[2] = a.data[0] * b.data[1] - b.data[0] * a.data[1];
    return ret;
  }
}
