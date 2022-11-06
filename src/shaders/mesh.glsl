#version 330

#if defined VERTEX_SHADER

in mat4 in_projection_matrix;
in mat4 in_view_matrix;
in vec3 in_position;
in vec4 in_color;
#if defined USE_UV
    in vec2 in_uv;
#endif
#if defined USE_MAP
    in sampler2D in_map;
#endif

out vec4 vert_color;
#if defined USE_UV
    out vec2 vert_uv;
#endif
#if defined USE_MAP
    out sampler2D vert_map;
#endif

void main() {
    vert_color = in_color;
    #if defined USE_UV
        vert_uv = in_uv;
    #endif
    #if defined USE_MAP
        vert_map = in_map;
    #endif
    gl_Position = in_projection_matrix * in_view_matrix * vec4(in_position, 1.0);
}

#elif defined FRAGMENT_SHADER

in vec4 vert_color;
#if defined USE_UV
    in vec2 vert_uv;
#endif
#if defined USE_MAP
    in sampler2D vert_map;
#endif

out vec4 frag_color;

void main() {
    frag_color = vert_color;
    #if defined USE_MAP
        frag_color *= texture(vert_map, vert_uv);
    #endif
}

#endif
