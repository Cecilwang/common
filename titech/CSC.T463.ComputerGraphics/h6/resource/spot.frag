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
    float ambient = 0.2;

    vec3  lightDir = normalize(lightpos - pos);
    float diffuse = max(dot(normal, lightDir), 0.0);

    vec3  cameraPos = vec3(2.,5.,-1.);
    vec3  cameraDir = normalize(cameraPos - pos);
    vec3 reflectDir = reflect(-lightDir, normal);
    float specular = 1.0 * pow(max(dot(cameraDir, reflectDir), 0.0), 32.);

    gl_FragColor = vec4((ambient+diffuse+specular)*texture2D(texture0, texcoord));
}
