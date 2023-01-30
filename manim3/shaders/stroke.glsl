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
    float u_width;
    vec4 u_color;
    float u_dilate;
};
layout (std140) uniform ub_winding_sign {
    float u_winding_sign;
};

const float PI = acos(-1.0);
const float PI_HALF = PI / 2.0;

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


in VS_GS {
    vec4 position;
} gs_in[];

out GS_FS {
    vec2 offset_vec;
} gs_out;


vec2 to_ndc_space(vec4 position) {
    return position.xy / position.w;
}


float get_direction_angle(vec4 position_0, vec4 position_1) {
    vec2 p0_ndc = to_ndc_space(position_0);
    vec2 p1_ndc = to_ndc_space(position_1);
    vec2 direction = frame_transform * (p1_ndc - p0_ndc);
    return atan(direction.y, direction.x);
}


void emit_vertex_by_polar(vec4 center_position, float magnitude, float angle) {
    vec2 offset_vec = magnitude * vec2(cos(angle), sin(angle));
    gs_out.offset_vec = offset_vec;
    gl_Position = center_position + vec4(frame_transform_inv * u_width * offset_vec, 0.0, 0.0);
    EmitVertex();
}


void emit_sector(vec4 center_position, float sector_middle_angle, float delta_angle) {
    float n_primitives = clamp(ceil(abs(delta_angle) / PI_HALF), 1.0, 2.0);
    float d_angle = delta_angle / (2.0 * n_primitives);
    float angle_start = sector_middle_angle - delta_angle / 2.0;
    for (int i = 0; i < n_primitives; ++i) {
        float angle_mid = angle_start + d_angle;
        float angle_end = angle_start + d_angle * 2.0;
        emit_vertex_by_polar(center_position, 0.0, 0.0);
        emit_vertex_by_polar(center_position, 1.0, angle_start);
        emit_vertex_by_polar(center_position, 1.0, angle_end);
        emit_vertex_by_polar(center_position, 1.0 / cos(d_angle), angle_mid);
        EndPrimitive();
        angle_start = angle_end;
    }
}


#if defined STROKE_LINE


layout (lines) in;
layout (triangle_strip, max_vertices = 6) out;


void both_sided(vec4 line_start_position, vec4 line_end_position, float direction_angle) {
    emit_vertex_by_polar(line_start_position, 1.0, direction_angle - PI_HALF);
    emit_vertex_by_polar(line_end_position, 1.0, direction_angle - PI_HALF);
    emit_vertex_by_polar(line_start_position, 0.0, 0.0);
    emit_vertex_by_polar(line_end_position, 0.0, 0.0);
    emit_vertex_by_polar(line_start_position, 1.0, direction_angle + PI_HALF);
    emit_vertex_by_polar(line_end_position, 1.0, direction_angle + PI_HALF);
    EndPrimitive();
}


void single_sided(vec4 line_start_position, vec4 line_end_position, float direction_angle) {
    emit_vertex_by_polar(line_start_position, 1.0, direction_angle - u_winding_sign * PI_HALF);
    emit_vertex_by_polar(line_end_position, 1.0, direction_angle - u_winding_sign * PI_HALF);
    emit_vertex_by_polar(line_start_position, 0.0, 0.0);
    emit_vertex_by_polar(line_end_position, 0.0, 0.0);
    EndPrimitive();
}


void main() {
    float direction_angle = get_direction_angle(gs_in[0].position, gs_in[1].position);
    line_subroutine(gs_in[0].position, gs_in[1].position, direction_angle);
}


#elif defined STROKE_JOIN


layout (triangles) in;
layout (triangle_strip, max_vertices = 8) out;


void both_sided(vec4 center_position, float direction_middle_angle, float delta_angle) {
    emit_sector(center_position, direction_middle_angle - sign(delta_angle) * PI_HALF, abs(delta_angle));
}


void single_sided(vec4 center_position, float direction_middle_angle, float delta_angle) {
    if (u_winding_sign * delta_angle > 0.0) {
        both_sided(center_position, direction_middle_angle, delta_angle);
    }
}


void main() {
    float direction_angle_0 = get_direction_angle(gs_in[0].position, gs_in[1].position);
    float direction_angle_1 = get_direction_angle(gs_in[1].position, gs_in[2].position);
    // -PI <= delta_angle < PI
    float delta_angle = mod(direction_angle_1 - direction_angle_0 + PI, 2.0 * PI) - PI;
    join_subroutine(gs_in[1].position, direction_angle_0 + delta_angle / 2.0, delta_angle);
}


#elif defined STROKE_CAP


layout (lines) in;
layout (triangle_strip, max_vertices = 8) out;


void main() {
    float opposite_direction_angle = get_direction_angle(gs_in[1].position, gs_in[0].position);
    emit_sector(gs_in[0].position, opposite_direction_angle, PI);
}


#elif defined STROKE_POINT


layout (points) in;
layout (triangle_strip, max_vertices = 16) out;


void main() {
    emit_sector(gs_in[0].position, 0.0, PI);
    emit_sector(gs_in[0].position, PI, PI);
}


#endif


/***************************/
#elif defined FRAGMENT_SHADER
/***************************/


in GS_FS {
    vec2 offset_vec;
} fs_in;

out vec4 frag_color;


void main() {
    float dilate_base = 1.0 - length(fs_in.offset_vec);
    if (dilate_base <= 0.0) {
        discard;
    }
    frag_color = vec4(u_color.rgb, u_color.a * pow(dilate_base, u_dilate));
}


#endif
