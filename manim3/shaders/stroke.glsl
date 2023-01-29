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

const float PI = 3.141592653589793;

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


vec2 get_unit_vector(float angle) {
    return vec2(cos(angle), sin(angle));
}


vec2 to_ndc_space(vec4 position) {
    return position.xy / position.w;
}


float get_direction_angle(vec4 position_0, vec4 position_1) {
    vec2 p0_ndc = to_ndc_space(position_0);
    vec2 p1_ndc = to_ndc_space(position_1);
    vec2 direction = frame_transform * (p1_ndc - p0_ndc);
    return atan(direction.y, direction.x);
}


void emit_vertex_by_offset_vec(vec4 center_position, vec2 offset_vec) {
    gs_out.offset_vec = offset_vec;
    gl_Position = center_position + vec4(frame_transform_inv * offset_vec * abs(u_width), 0.0, 0.0);
    EmitVertex();
}


void emit_sector(vec4 center_position, float angle_start, float delta_angle, float width_sign) {
    float n_primitives = clamp(ceil(abs(delta_angle) / PI * 2.0), 1.0, 2.0);
    float d_angle = delta_angle / (2.0 * n_primitives);
    for (int i = 0; i < n_primitives; ++i) {
        float angle_mid = angle_start + d_angle;
        float angle_end = angle_start + d_angle * 2.0;
        emit_vertex_by_offset_vec(center_position, vec2(0.0));
        emit_vertex_by_offset_vec(center_position, width_sign * get_unit_vector(angle_start));
        emit_vertex_by_offset_vec(center_position, width_sign * get_unit_vector(angle_end));
        emit_vertex_by_offset_vec(center_position, width_sign / cos(d_angle) * get_unit_vector(angle_mid));
        EndPrimitive();

        angle_start = angle_end;
    }
}


#if defined STROKE_LINE


layout (lines) in;
layout (triangle_strip, max_vertices = 6) out;


void single_sided(vec4 line_start_position, vec4 line_end_position, vec2 offset_vec) {
    emit_vertex_by_offset_vec(line_start_position, offset_vec);
    emit_vertex_by_offset_vec(line_end_position, offset_vec);
    emit_vertex_by_offset_vec(line_start_position, vec2(0.0));
    emit_vertex_by_offset_vec(line_end_position, vec2(0.0));
}


void both_sided(vec4 line_start_position, vec4 line_end_position, vec2 offset_vec) {
    single_sided(line_start_position, line_end_position, offset_vec);
    emit_vertex_by_offset_vec(line_start_position, -offset_vec);
    emit_vertex_by_offset_vec(line_end_position, -offset_vec);
}


void main() {
    float direction_angle = get_direction_angle(gs_in[0].position, gs_in[1].position);
    vec2 offset_vec = u_winding_sign * sign(u_width) * get_unit_vector(direction_angle - PI / 2.0);
    line_subroutine(gs_in[0].position, gs_in[1].position, offset_vec);
    EndPrimitive();
}


#elif defined STROKE_JOIN


layout (triangles) in;
layout (triangle_strip, max_vertices = 8) out;


void single_sided(vec4 center_position, float angle_start, float delta_angle) {
    if (delta_angle * u_width * u_winding_sign > 0.0) {
        emit_sector(center_position, angle_start, delta_angle, sign(u_width) * u_winding_sign);
    }
}


void both_sided(vec4 center_position, float angle_start, float delta_angle) {
    emit_sector(center_position, angle_start, delta_angle, sign(delta_angle));
}


void main() {
    float direction_angle_0 = get_direction_angle(gs_in[0].position, gs_in[1].position);
    float direction_angle_1 = get_direction_angle(gs_in[1].position, gs_in[2].position);
    // -PI <= delta_angle < PI
    float delta_angle = mod(direction_angle_1 - direction_angle_0 + PI, 2.0 * PI) - PI;
    join_subroutine(gs_in[1].position, direction_angle_0 - PI / 2.0, delta_angle);
}


#elif defined STROKE_CAP


layout (lines) in;
layout (triangle_strip, max_vertices = 8) out;


void main() {
    float direction_angle = get_direction_angle(gs_in[0].position, gs_in[1].position);
    emit_sector(gs_in[0].position, direction_angle + PI / 2.0, PI, 1.0);
}


#elif defined STROKE_POINT


layout (points) in;
layout (triangle_strip, max_vertices = 16) out;


void main() {
    emit_sector(gs_in[0].position, 0.0, PI, 1.0);
    emit_sector(gs_in[0].position, 0.0, -PI, 1.0);
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
    float distance_to_edge = 1.0 - length(fs_in.offset_vec);
    if (distance_to_edge <= 0.0) {
        discard;
    }
    frag_color = vec4(u_color.rgb, u_color.a * pow(distance_to_edge, u_dilate));
}


#endif
