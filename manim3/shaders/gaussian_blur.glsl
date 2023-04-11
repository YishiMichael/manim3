uniform sampler2D t_color_map;

layout (std140) uniform ub_gaussian_blur {
    vec2 u_uv_offset;
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


vec2 horizontal_dilate() {
    return vec2(u_uv_offset.x, 0.0);
}


vec2 vertical_dilate() {
    return vec2(0.0, u_uv_offset.y);
}


void main() {
    vec2 uv = fs_in.uv;
    vec2 directional_offset = blur_subroutine();
    frag_color = texture(t_color_map, uv) * u_convolution_core[0];
    for (int i = 1; i < CONVOLUTION_CORE_SIZE; ++i) {
        frag_color += texture(t_color_map, uv + i * directional_offset) * u_convolution_core[i];
        frag_color += texture(t_color_map, uv - i * directional_offset) * u_convolution_core[i];
    }
}


#endif
