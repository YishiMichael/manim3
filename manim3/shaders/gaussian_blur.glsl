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


vec4 horizontal_dilate(vec2 uv, sampler2D image) {
    vec2 offset = 1.0 / textureSize(image, 0);
    vec4 result = texture(image, uv) * u_convolution_core[0];
    for (int i = 1; i < CONVOLUTION_CORE_SIZE; ++i) {
        result += texture(image, uv + vec2(offset.x * i, 0.0)) * u_convolution_core[i];
        result += texture(image, uv - vec2(offset.x * i, 0.0)) * u_convolution_core[i];
    }
    return result;
}


vec4 vertical_dilate(vec2 uv, sampler2D image) {
    vec2 offset = 1.0 / textureSize(image, 0);
    vec4 result = texture(image, uv) * u_convolution_core[0];
    for (int i = 1; i < CONVOLUTION_CORE_SIZE; ++i) {
        result += texture(image, uv + vec2(0.0, offset.y * i)) * u_convolution_core[i];
        result += texture(image, uv - vec2(0.0, offset.y * i)) * u_convolution_core[i];
    }
    return result;
}


void main() {
    frag_color = blur_subroutine(fs_in.uv, u_color_map);
}


#endif
