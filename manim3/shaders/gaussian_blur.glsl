uniform sampler2D u_color_map;

layout (std140) uniform ub_convolution_core {
    float u_convolution_core[CONVOLUTION_CORE_SIZE];
};


/***********************/
#if defined VERTEX_SHADER
/***********************/


in vec3 in_position;
in vec2 in_uv;

out VS_FS {
    vec2 uv;
} vs_out;


void main() {
    vs_out.uv = in_uv;
    gl_Position = vec4(in_position, 1.0);
}


/***************************/
#elif defined FRAGMENT_SHADER
/***************************/


in VS_FS {
    vec2 uv;
} fs_in;

out vec4 frag_color;


vec2 horizontal_dilate(vec2 offset) {
    return vec2(offset.x, 0.0);
}


vec2 vertical_dilate(vec2 offset) {
    return vec2(0.0, offset.y);
}


void main() {
    vec2 uv = fs_in.uv;
    vec2 offset = 1.0 / textureSize(u_color_map, 0);
    vec2 directional_offset = blur_subroutine(offset);
    frag_color = texture(u_color_map, uv) * u_convolution_core[0];
    for (int i = 1; i < CONVOLUTION_CORE_SIZE; ++i) {
        frag_color += texture(u_color_map, uv + i * directional_offset) * u_convolution_core[i];
        frag_color += texture(u_color_map, uv - i * directional_offset) * u_convolution_core[i];
    }
}


#endif
