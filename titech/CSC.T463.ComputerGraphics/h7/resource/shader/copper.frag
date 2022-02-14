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
  vec3 n = normalize(normal);
  vec3 r = reflect(-l, n);
  vec3 v = normalize(viewdir);

  vec3 ambient = vec3(0.19125, 0.0735, 0.0225);
  vec3 diffuse = max(dot(l, n), 0.0) * vec3(0.7038, 0.27048, 0.0828);
  vec3 spec = pow(max(dot(r, v), 0.0), 12.8) * vec3(0.256777, 0.137622, 0.086014);

  gl_FragColor = vec4(ambient+diffuse+spec, 1.);
}
