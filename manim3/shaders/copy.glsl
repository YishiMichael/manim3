#version 430 core


#if NUM_U_COLOR_MAPS
uniform sampler2D u_color_maps[NUM_U_COLOR_MAPS];
#endif
#if NUM_U_DEPTH_MAPS
uniform sampler2D u_depth_maps[NUM_U_DEPTH_MAPS];
#endif


#if defined VERTEX_SHADER


in vec3 in_position;
in vec2 in_uv;

out VS_FS {
    vec2 uv;
} vs_out;

void main() {
    vs_out.uv = in_uv;
    gl_Position = vec4(in_position, 1.0);
}


#elif defined FRAGMENT_SHADER


in VS_FS {
    vec2 uv;
} fs_in;

#if NUM_U_COLOR_MAPS
out vec4 frag_color[NUM_U_COLOR_MAPS];
#endif

void main() {
    #if NUM_U_COLOR_MAPS
    for (int i = 0; i < NUM_U_COLOR_MAPS; ++i) {
        frag_color[i] = texture(u_color_maps[i], fs_in.uv);
    }
    #endif
    #if NUM_U_DEPTH_MAPS
    for (int i = 0; i < NUM_U_DEPTH_MAPS; ++i) {
        gl_FragDepth += texture(u_depth_maps[i], fs_in.uv).x;
    }
    #endif
}


#endif
