//#version 120
//
// simple.frag
//
uniform mat4 mat[4];
uniform sampler2D texture0;
uniform vec3 lightpos;
varying vec3 normal;
varying vec4 color;
varying vec2 texcoord;
varying vec3 pos;

void main (void){
    float ambient = 0.1;

    vec3  lightDir = normalize(lightpos - pos);
    float diffuse = 0.5 * max(dot(normal, lightDir), 0.0);

    gl_FragColor = vec4((ambient+diffuse)*texture2D(texture0, texcoord));
}
