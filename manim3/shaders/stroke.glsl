layout (std140) uniform ub_camera {
    mat4 u_projection_view_matrix;
    vec3 u_view_position;
    vec2 u_frame_radii;
};
layout (std140) uniform ub_model {
    mat4 u_model_matrix;
};
layout (std140) uniform ub_stroke {
    vec4 u_color;
    float u_width;
    float u_dilate;
};
//layout (std140) uniform ub_winding_sign {
//    float u_winding_sign;
//};

const float PI_HALF = acos(0.0);
const float PI = PI_HALF * 2.0;


/***********************/
#if defined VERTEX_SHADER
/***********************/


in vec3 in_position;
//in float in_direction_angle;
//in float in_delta_angle;

out VS_GS {
    vec4 view_position;
    //float direction_angle;
    //float delta_angle;
} vs_out;


void main() {
    vs_out.view_position = u_projection_view_matrix * u_model_matrix * vec4(in_position, 1.0);
    //vs_out.direction_angle = in_direction_angle;
    //vs_out.delta_angle = in_delta_angle;
}


/***************************/
#elif defined GEOMETRY_SHADER
/***************************/


// For every point `p`, `direction_angle` and `delta_angle` satisfies
// `polar(direction_angle - delta_angle) = normalize(p - p_prev)`,
// `polar(direction_angle + delta_angle) = normalize(p_next - p)`,
// `-PI_HALF < delta_angle < PI_HALF`.
in VS_GS {
    vec4 view_position;
    //float direction_angle;
    //float delta_angle;
} gs_in[];

out GS_FS {
    vec2 offset_vec;
} gs_out;


layout (lines_adjacency) in;
layout (triangle_strip, max_vertices = 16) out;


const float width = abs(u_width);
//const float winding_sign = sign(u_width) * u_winding_sign;  // Requires `u_width != 0.0`.
const float width_sign = sign(u_width);  // Requires `u_width != 0.0`.


float get_angle(vec3 vector) {
    // Returns an angle in `[-PI, PI]`
    return atan(vector.y, vector.x);
}


vec3 get_position(vec4 view_position) {
    return view_position.xyz / view_position.w * vec3(u_frame_radii, 1.0);
}


void emit_vertex_by_polar(vec3 center_position, float magnitude, float angle) {
    vec2 offset_vec = magnitude * vec2(cos(angle), sin(angle));
    gs_out.offset_vec = offset_vec;
    gl_Position = vec4((center_position + vec3(width * offset_vec, 0.0)) / vec3(u_frame_radii, 1.0), 1.0);
    EmitVertex();
}


//void emit_sector(vec3 center_position, float sector_middle_angle, float delta_angle) {
//    // Emit a diamond-like shape covering the sector.
//    // `delta_angle` is intepreted as half the radius angle.
//    float d_angle = delta_angle / 2.0;
//    emit_vertex_by_polar(center_position, 0.0, sector_middle_angle);
//    emit_vertex_by_polar(center_position, 1.0, sector_middle_angle - delta_angle);
//    emit_vertex_by_polar(center_position, 1.0, sector_middle_angle);
//    emit_vertex_by_polar(center_position, 1.0 / cos(d_angle), sector_middle_angle - d_angle);
//    EndPrimitive();
//    emit_vertex_by_polar(center_position, 0.0, sector_middle_angle);
//    emit_vertex_by_polar(center_position, 1.0, sector_middle_angle);
//    emit_vertex_by_polar(center_position, 1.0, sector_middle_angle + delta_angle);
//    emit_vertex_by_polar(center_position, 1.0 / cos(d_angle), sector_middle_angle + d_angle);
//    EndPrimitive();
//}


//#if defined STROKE_LINE


float clamp_angle(float diff_angle) {
    // `[-2 PI, 2 PI] -> (-PI, PI]`
    if (diff_angle > PI) {
        return diff_angle - PI * 2.0;
    } else if (diff_angle <= -PI) {
        return diff_angle + PI * 2.0;
    } else {
        return diff_angle;
    }
}


void emit_sector_quad(vec3 center_position, float start_angle, float sweep_angle) {
    // Requires `|sweep_angle| <= PI / 2`
    float half_sweep_angle = sweep_angle / 2.0;
    float sector_middle_angle = start_angle + half_sweep_angle;
    emit_vertex_by_polar(center_position, 0.0, sector_middle_angle);
    emit_vertex_by_polar(center_position, 1.0, sector_middle_angle - half_sweep_angle);
    emit_vertex_by_polar(center_position, 1.0, sector_middle_angle + half_sweep_angle);
    emit_vertex_by_polar(center_position, 1.0 / cos(half_sweep_angle), sector_middle_angle);
    EndPrimitive();
}


void emit_single_dot(vec3 center_position) {
    emit_sector_quad(center_position, 0.0, PI_HALF);
    emit_sector_quad(center_position, PI_HALF, PI_HALF);
    emit_sector_quad(center_position, PI, PI_HALF);
    emit_sector_quad(center_position, -PI_HALF, PI_HALF);
}


void emit_one_side(vec3 position_0, vec3 position_1, float line_length, float normal_angle, float diff_angle_0, float diff_angle_1) {
    //float line_length = length(position_1 - position_0);
    float delta_angle_0 = clamp_angle(diff_angle_0) / 2.0;
    float delta_angle_1 = clamp_angle(diff_angle_1) / 2.0;
    //if (delta_angle_0 > PI_HALF) {
    //    delta_angle_0 -= PI;
    //} else if (delta_angle_0 <= -PI_HALF) {
    //    delta_angle_0 += PI;
    //}
    //if (delta_angle_1 > PI_HALF) {
    //    delta_angle_1 -= PI;
    //} else if (delta_angle_1 <= -PI_HALF) {
    //    delta_angle_1 += PI;
    //}
    //delta_angle_0 = clamp_delta_angle(delta_angle_0);
    //delta_angle_1 = clamp_delta_angle(delta_angle_1);

    float ratio_0 = delta_angle_0 < 0.0 ? tan(-delta_angle_0) * width / line_length : 0.0;
    float ratio_1 = delta_angle_1 < 0.0 ? tan(-delta_angle_1) * width / line_length : 0.0;
    float ratio_sum = ratio_0 + ratio_1;
    if (ratio_sum > 1.0) {
        emit_vertex_by_polar(position_0, 0.0, normal_angle);
        emit_vertex_by_polar(position_1, 0.0, normal_angle);
        emit_vertex_by_polar(
            mix(position_0, position_1, ratio_0 / ratio_sum),
            1.0 / ratio_sum,
            normal_angle
        );
        EndPrimitive();
    } else {
        emit_vertex_by_polar(position_0, 0.0, normal_angle);
        emit_vertex_by_polar(position_1, 0.0, normal_angle);
        emit_vertex_by_polar(mix(position_0, position_1, ratio_0), 1.0, normal_angle);
        emit_vertex_by_polar(mix(position_1, position_0, ratio_1), 1.0, normal_angle);
        EndPrimitive();
    }

    if (delta_angle_0 > 0.0) {
        emit_sector_quad(position_0, normal_angle, -delta_angle_0);
    }
    if (delta_angle_1 > 0.0) {
        emit_sector_quad(position_1, normal_angle, delta_angle_1);
    }
}


void both_sided(vec3 position_0, vec3 position_1, float line_length, float line_angle, float diff_angle_0, float diff_angle_1) {
    emit_one_side(
        position_0, position_1, line_length,
        line_angle - PI_HALF, diff_angle_0, diff_angle_1
    );
    emit_one_side(
        position_0, position_1, line_length,
        line_angle + PI_HALF, -diff_angle_0, -diff_angle_1
    );
}


void single_sided(vec3 position_0, vec3 position_1, float line_length, float line_angle, float diff_angle_0, float diff_angle_1) {
    emit_one_side(
        position_0, position_1, line_length,
        line_angle - width_sign * PI_HALF, width_sign * diff_angle_0, width_sign * diff_angle_1
    );
}


void main() {
    if (width == 0.0) {
        return;
    }

    vec3 position_prev = get_position(gs_in[0].view_position);
    vec3 position_0 = get_position(gs_in[1].view_position);
    vec3 position_1 = get_position(gs_in[2].view_position);
    vec3 position_next = get_position(gs_in[3].view_position);

    vec3 vector_prev = position_0 - position_prev;
    vec3 vector = position_1 - position_0;
    vec3 vector_next = position_next - position_1;
    float vector_prev_length = length(vector_prev);
    float vector_length = length(vector);
    float vector_next_length = length(vector_next);

    //float angle_01 = get_angle(position_1 - position_0);
    //float angle_12 = get_angle(position_2 - position_1);
    //float angle_23 = get_angle(position_3 - position_2);

    if (vector_length == 0.0) {
        if (vector_prev_length != 0.0 || vector_next_length != 0.0) {
            return;
        }
        emit_single_dot(position_0);
        return;
    }

    float line_angle = get_angle(vector);
    // Default to `PI` to form caps at both endpoints.
    float diff_angle_0 = vector_prev_length != 0.0 ? line_angle - get_angle(vector_prev) : PI;
    float diff_angle_1 = vector_next_length != 0.0 ? get_angle(vector_next) - line_angle : PI;
    stroke_subroutine(position_0, position_1, vector_length, line_angle, diff_angle_0, diff_angle_1);
    //float line_angle = gs_in[0].direction_angle + gs_in[0].delta_angle;
    //line_subroutine(gs_in[0].position, gs_in[1].position, gs_in[0].delta_angle, gs_in[1].delta_angle, line_angle);
}


//#elif defined STROKE_JOIN


//layout (points) in;
//layout (triangle_strip, max_vertices = 8) out;


//void both_sided(vec3 position, float direction_angle, float delta_angle) {
//    if (delta_angle == 0.0) {
//        return;
//    }
//    float sector_middle_angle = direction_angle - sign(delta_angle) * PI_HALF;
//    emit_sector(position, sector_middle_angle, abs(delta_angle));
//}


//void single_sided(vec3 position, float direction_angle, float delta_angle) {
//    if (width_sign * delta_angle > 0.0) {
//        both_sided(position, direction_angle, delta_angle);
//    }
//}


//void main() {
//    join_subroutine(gs_in[0].position, gs_in[0].direction_angle, gs_in[0].delta_angle);
//}


//#elif defined STROKE_CAP


//layout (lines) in;
//layout (triangle_strip, max_vertices = 16) out;


//void main() {
//    emit_sector(gs_in[0].position, gs_in[0].direction_angle + PI, PI_HALF);
//    emit_sector(gs_in[1].position, gs_in[1].direction_angle, PI_HALF);
//}


//#endif


/***************************/
#elif defined FRAGMENT_SHADER
/***************************/


in GS_FS {
    vec2 offset_vec;
} fs_in;

#if defined IS_TRANSPARENT
out vec4 frag_accum;
out float frag_revealage;
#else
out vec4 frag_color;
#endif


void main() {
    float dilate_base = 1.0 - length(fs_in.offset_vec);
    if (dilate_base <= 0.0) {
        discard;
    }
    vec4 color = u_color;
    color.a *= pow(dilate_base, u_dilate);

    #if defined IS_TRANSPARENT
    frag_accum = color;
    frag_accum.rgb *= color.a;
    frag_revealage = color.a;
    #else
    frag_color = color;
    #endif
}


#endif
