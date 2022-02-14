//#version 120
//
// simple.vert
//
//invariant gl_Position; // invariant out gl_Position; //for #version 130
attribute vec3 inposition;//in vec3 position;          //for #version 130
attribute vec4 incolor;
attribute vec3 innormal;
attribute vec2 intexcoord0;
attribute vec3 intangent;
varying vec3 normal;
varying vec4 color;
varying vec2 texcoord;
varying vec3 viewdir;
varying vec3 lightdir;
uniform mat4 mat[4];
uniform vec3 lightpos;

void main(void)
{
  normal = normalize((mat[3]*vec4(innormal,1.0)).xyz);
  vec3 tangent = normalize((mat[3]*vec4(intangent,1.0)).xyz);
  vec3 bitangent = cross(normal, tangent);
  vec3 position = (mat[1]*vec4(inposition,1.0)).xyz;
  //  vec3 tmplightpos = (mat[1]*vec4(lightpos,1.0)).xyz;
  vec3 tmplightpos = lightpos;
  viewdir = -vec3(dot(position, tangent),
                 dot(position, bitangent),
                 dot(position, normal));
  vec3 tmplightdir = normalize(tmplightpos-position);
  lightdir = vec3(dot(tmplightdir, tangent),
                  dot(tmplightdir, bitangent),
                  dot(tmplightdir, normal));
  color = incolor;
  texcoord = intexcoord0;
  gl_Position = mat[0]*mat[1]*vec4(inposition, 1.0);
//  gl_Position = vec4(inposition, 1.0);
//  position = gl_Position.xyz;
}
