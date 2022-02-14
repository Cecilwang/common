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
varying vec3 pos;
uniform mat4 mat[4];


void main(void)
{
  // normalize(Mvit * normal)
  normal = normalize((mat[3]*vec4(innormal,0.0)).xyz);
  color = incolor;
  // ref: https://www.alexisgiard.com/icosahedron-sphere/
  texcoord.x = atan(inposition.z, inposition.x) / (2. *  3.14159265358979323);
  texcoord.y = asin(inposition.y) / 3.14159265358979323 + 0.5;
  gl_Position = mat[0]*mat[1]*vec4(inposition, 1.0);
  pos = gl_Position.xyz;
}

