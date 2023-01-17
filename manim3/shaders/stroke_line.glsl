#version 430 core


subroutine void line_dilate_func_t(vec4 line_start_gl_position, vec4 line_end_gl_position, vec4 offset_vec);

layout (std140) uniform ub_camera {
    mat4 u_projection_matrix;
    mat4 u_view_matrix;
    vec3 u_view_position;
    vec2 u_frame_radius;
};
layout (std140) uniform ub_model {
    mat4 u_model_matrix;
};
layout (std140) uniform ub_stroke {
    float u_stroke_width;
    vec4 u_stroke_color;
    float u_stroke_dilate;
};

subroutine uniform line_dilate_func_t line_dilate_func;

const mat2 frame_transform = mat2(
    u_frame_radius.x, 0.0,
    0.0, u_frame_radius.y
);
const mat2 frame_transform_inv = mat2(
    1.0 / u_frame_radius.x, 0.0,
    0.0, 1.0 / u_frame_radius.y
);


/***********************/
#if defined VERTEX_SHADER
/***********************/


in vec3 in_position;

out VS_GS {
    vec4 gl_position;
} vs_out;


void main() {
    vs_out.gl_position = u_projection_matrix * u_view_matrix * u_model_matrix * vec4(in_position, 1.0);
}


/***************************/
#elif defined GEOMETRY_SHADER
/***************************/


layout (lines) in;
layout (triangle_strip, max_vertices = 6) out;

in VS_GS {
    vec4 gl_position;
} gs_in[2];

out GS_FS {
    float dilate_factor;
} gs_out;


subroutine(line_dilate_func_t)
void single_sided_dilate(vec4 line_start_gl_position, vec4 line_end_gl_position, vec4 offset_vec) {
    gs_out.dilate_factor = 0.0;
    gl_Position = line_start_gl_position + offset_vec;
    EmitVertex();
    gl_Position = line_end_gl_position + offset_vec;
    EmitVertex();
    gs_out.dilate_factor = 1.0;
    gl_Position = line_start_gl_position;
    EmitVertex();
    gl_Position = line_end_gl_position;
    EmitVertex();
}


subroutine(line_dilate_func_t)
void both_sided_dilate(vec4 line_start_gl_position, vec4 line_end_gl_position, vec4 offset_vec) {
    single_sided_dilate(line_start_gl_position, line_end_gl_position, offset_vec);
    gs_out.dilate_factor = 0.0;
    gl_Position = line_start_gl_position - offset_vec;
    EmitVertex();
    gl_Position = line_end_gl_position - offset_vec;
    EmitVertex();
}


void main() {
    vec4 line_start_gl_position = gs_in[0].gl_position;
    vec4 line_end_gl_position = gs_in[1].gl_position;
    vec2 direction = vec2(line_end_gl_position - line_start_gl_position);
    // Rotate 90 degrees counterclockwise, while taking the aspect ratio into consideration
    vec2 transformed_direction = frame_transform * direction;
    vec2 transformed_normal = normalize(vec2(-transformed_direction.y, transformed_direction.x));
    vec2 normal = frame_transform_inv * transformed_normal;
    vec4 offset_vec = u_stroke_width * vec4(normal, 0.0, 0.0);

    line_dilate_func(line_start_gl_position, line_end_gl_position, offset_vec);
    EndPrimitive();
}


/***************************/
#elif defined FRAGMENT_SHADER
/***************************/


in GS_FS {
    float dilate_factor;
} fs_in;

out vec4 frag_color;


void main() {
    frag_color = vec4(u_stroke_color.rgb, u_stroke_color.a * pow(fs_in.dilate_factor, u_stroke_dilate));
}


#endif
