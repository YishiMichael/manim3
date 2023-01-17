#version 430 core


subroutine void join_dilate_func_t(vec4 center_position, float angle_start, float delta_angle);

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

subroutine uniform join_dilate_func_t join_dilate_func;

const float PI = 3.141592653589793;
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


layout (triangles) in;
layout (triangle_strip, max_vertices = 8) out;

in VS_GS {
    vec4 gl_position;
} gs_in[3];

out GS_FS {
    vec2 transformed_offset_vec;
} gs_out;


float get_normal_angle(vec2 direction) {
    return atan(direction.x, -direction.y);
}


vec2 get_unit_vector(float angle) {
    return vec2(cos(angle), sin(angle));
}


void emit_vertex_by_transformed_offset_vec(vec4 center_position, vec2 transformed_offset_vec) {
    gs_out.transformed_offset_vec = transformed_offset_vec;
    gl_Position = center_position + vec4(frame_transform_inv * transformed_offset_vec, 0.0, 0.0);
    EmitVertex();
}


void emit_sector(float stroke_width, vec4 center_position, float angle_start, float delta_angle) {
    float n_primitives = clamp(ceil(abs(delta_angle) / PI * 2.0), 1.0, 2.0);
    float d_angle = delta_angle / (2.0 * n_primitives);
    for (int i = 0; i < n_primitives; ++i) {
        float angle_mid = angle_start + d_angle;
        float angle_end = angle_start + d_angle * 2.0;
        emit_vertex_by_transformed_offset_vec(center_position, vec2(0.0));
        emit_vertex_by_transformed_offset_vec(center_position, stroke_width * get_unit_vector(angle_start));
        emit_vertex_by_transformed_offset_vec(center_position, stroke_width * get_unit_vector(angle_end));
        emit_vertex_by_transformed_offset_vec(center_position, stroke_width / cos(d_angle) * get_unit_vector(angle_mid));
        EndPrimitive();

        angle_start = angle_end;
    }
}


subroutine(join_dilate_func_t)
void single_sided_dilate(vec4 center_position, float angle_start, float delta_angle) {
    if (delta_angle * u_stroke_width < 0.0) {
        emit_sector(u_stroke_width, center_position, angle_start, delta_angle);
    }
}


subroutine(join_dilate_func_t)
void both_sided_dilate(vec4 center_position, float angle_start, float delta_angle) {
    emit_sector(-sign(delta_angle) * abs(u_stroke_width), center_position, angle_start, delta_angle);
}


void main() {
    vec4 center_position = gs_in[1].gl_position;
    float normal_angle_0 = get_normal_angle(frame_transform * vec2(gs_in[1].gl_position - gs_in[0].gl_position));
    float normal_angle_1 = get_normal_angle(frame_transform * vec2(gs_in[2].gl_position - gs_in[1].gl_position));
    // -PI <= delta_angle < PI
    float delta_angle = mod(normal_angle_1 - normal_angle_0 + PI, 2.0 * PI) - PI;

    join_dilate_func(center_position, normal_angle_0, delta_angle);
}


/***************************/
#elif defined FRAGMENT_SHADER
/***************************/


in GS_FS {
    vec2 transformed_offset_vec;
} fs_in;

out vec4 frag_color;


void main() {
    float dilate_factor = 1.0 - length(fs_in.transformed_offset_vec) / abs(u_stroke_width);
    if (dilate_factor <= 0.0) {
        discard;
    }
    frag_color = vec4(u_stroke_color.rgb, u_stroke_color.a * pow(dilate_factor, u_stroke_dilate));
}


#endif
