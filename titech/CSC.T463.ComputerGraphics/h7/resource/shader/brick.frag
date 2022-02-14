//#version 120
//
// simple.frag
//
uniform mat4 mat[4];
uniform sampler2D texture0;
uniform vec3 lightpos;
uniform vec3 lightcolor;
varying vec3 normal;
varying vec4 color;
varying vec2 texcoord;
varying vec3 viewdir;
varying vec3 lightdir;

void main (void){
  vec3 l = normalize(lightdir);
  vec3 bump = (texture2D(texture0, texcoord)*2.0-1.0).xyz;
  vec3 n = normalize(normal+0.15*bump);
  //vec3 n = normalize(normal);
  vec3 r = reflect(-l, n);
  vec3 v = normalize(viewdir);

  float diffuse   = max(dot(l, n), 0.0);
  float spec = max(dot(r, v), 0.0);
  spec = pow(spec, 16.0);
  //gl_FragColor = (0.7 * diffuse + 0.3 * spec)*mix(texture2D(texture0, texcoord), color, 0.8);
  gl_FragColor = vec4((0.1 + 0.7 * diffuse + 0.3 * spec)*color.xyz, 1.);
}
