#version 330


#if defined VERTEX_SHADER


in mat4 in_projection_matrix;
in mat4 in_view_matrix;
//in mat4 in_model_matrix;
in vec3 in_position;
in vec4 in_color;
#if defined USE_UV
    in vec2 in_uv;
#endif

out vec4 vert_color;
#if defined USE_UV
    out vec2 vert_uv;
#endif

void main() {
    vert_color = in_color;
    #if defined USE_UV
        vert_uv = in_uv;
    #endif
    gl_Position = in_projection_matrix * in_view_matrix * vec4(in_position, 1.0);
}


#elif defined FRAGMENT_SHADER


#if defined USE_COLOR_MAP
    uniform sampler2D uniform_color_map;
#endif

in vec4 vert_color;
#if defined USE_UV
    in vec2 vert_uv;
#endif

out vec4 frag_color;

void main() {
    frag_color = vert_color;
    #if defined USE_COLOR_MAP
        frag_color *= texture(uniform_color_map, vert_uv);
    #endif
}


#endif
