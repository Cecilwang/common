//#version 120
//
// simple.vert
//
//invariant gl_Position; // invariant out gl_Position; //for #version 130
attribute vec3 inposition;//in vec3 position;          //for #version 130
attribute vec4 incolor;
attribute vec3 innormal;
attribute vec2 intexcoord0;
varying vec3 normal;
varying vec4 color;
varying vec2 texcoord;
uniform mat4 mat[4];
 
void main(void)
{
  normal = normalize((mat[3]*vec4(innormal,1.0)).xyz);
  color = incolor;
  texcoord = intexcoord0;
  gl_Position = mat[0]*mat[1]*vec4(inposition, 1.0);
//  gl_Position = vec4(inposition, 1.0);

}

