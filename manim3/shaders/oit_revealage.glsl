//uniform sampler2D u_color_map;
//uniform sampler2D u_depth_map;


///***********************/
//#if defined VERTEX_SHADER
///***********************/


//in vec3 in_position;
//in vec2 in_uv;

//out VS_FS {
//    vec2 uv;
//} vs_out;


//void main() {
//    vs_out.uv = in_uv;
//    gl_Position = vec4(in_position, 1.0);
//}


///***************************/
//#elif defined FRAGMENT_SHADER
///***************************/


//in VS_FS {
//    vec2 uv;
//} fs_in;

//out float frag_revealage;


//void main() {
//    frag_revealage = texture(u_color_map, fs_in.uv).a;
//    gl_FragDepth = texture(u_depth_map, fs_in.uv).x;
//}


//#endif
