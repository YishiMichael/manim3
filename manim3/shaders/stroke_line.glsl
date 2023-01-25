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

mat2 frame_transform = mat2(
    u_frame_radius.x, 0.0,
    0.0, u_frame_radius.y
);
mat2 frame_transform_inv = mat2(
    1.0 / u_frame_radius.x, 0.0,
    0.0, 1.0 / u_frame_radius.y
);


/***********************/
#if defined VERTEX_SHADER
/***********************/


in vec3 in_position;

out VS_GS {
    vec4 position;
} vs_out;


void main() {
    vs_out.position = u_projection_matrix * u_view_matrix * u_model_matrix * vec4(in_position, 1.0);
}


/***************************/
#elif defined GEOMETRY_SHADER
/***************************/


layout (lines) in;
layout (triangle_strip, max_vertices = 6) out;

in VS_GS {
    vec4 position;
} gs_in[2];

out GS_FS {
    float distance_to_edge;
} gs_out;


vec2 to_ndc_space(vec4 position) {
    return position.xy / position.w;
}


void single_sided(vec4 line_start_position, vec4 line_end_position, vec4 offset_vec) {
    gs_out.distance_to_edge = 0.0;
    gl_Position = line_start_position + offset_vec;
    EmitVertex();
    gl_Position = line_end_position + offset_vec;
    EmitVertex();
    gs_out.distance_to_edge = 1.0;
    gl_Position = line_start_position;
    EmitVertex();
    gl_Position = line_end_position;
    EmitVertex();
}


void both_sided(vec4 line_start_position, vec4 line_end_position, vec4 offset_vec) {
    single_sided(line_start_position, line_end_position, offset_vec);
    gs_out.distance_to_edge = 0.0;
    gl_Position = line_start_position - offset_vec;
    EmitVertex();
    gl_Position = line_end_position - offset_vec;
    EmitVertex();
}


void main() {
    vec2 p0_ndc = to_ndc_space(gs_in[0].position);
    vec2 p1_ndc = to_ndc_space(gs_in[1].position);
    // Rotate 90 degrees counterclockwise, while taking the aspect ratio into consideration
    vec2 transformed_direction = frame_transform * (p1_ndc - p0_ndc);
    vec2 transformed_normal = normalize(vec2(-transformed_direction.y, transformed_direction.x));
    vec2 normal = frame_transform_inv * transformed_normal;
    vec4 offset_vec = u_stroke_width * vec4(normal, 0.0, 0.0);

    line_subroutine(gs_in[0].position, gs_in[1].position, offset_vec);
    EndPrimitive();
}


/***************************/
#elif defined FRAGMENT_SHADER
/***************************/


in GS_FS {
    float distance_to_edge;
} fs_in;

out vec4 frag_color;


void main() {
    float distance_to_edge = fs_in.distance_to_edge;
    frag_color = vec4(u_stroke_color.rgb, u_stroke_color.a * pow(distance_to_edge, u_stroke_dilate));
}


#endif
