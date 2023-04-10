layout (std140) uniform ub_camera {
    mat4 u_projection_matrix;
    mat4 u_view_matrix;
    vec3 u_view_position;
    vec2 u_frame_radius;
};
//layout (std140) uniform ub_model {
//    mat4 u_model_matrix;
//};
layout (std140) uniform ub_stroke {
    float u_width;
    vec4 u_color;
    float u_dilate;
};
layout (std140) uniform ub_winding_sign {
    float u_winding_sign;
};

const float PI_HALF = acos(0.0);
const float PI = PI_HALF * 2.0;


/***********************/
#if defined VERTEX_SHADER
/***********************/


in vec3 in_position;
in float in_direction_angle;
in float in_delta_angle;

out VS_GS {
    vec3 position;
    float direction_angle;
    float delta_angle;
} vs_out;


void main() {
    vs_out.position = in_position;
    vs_out.direction_angle = in_direction_angle;
    vs_out.delta_angle = in_delta_angle;
}


/***************************/
#elif defined GEOMETRY_SHADER
/***************************/


// For every point `p`, `direction_angle` and `delta_angle` satisfies
// `polar(direction_angle - delta_angle) = normalize(p - p_prev)`
// `polar(direction_angle + delta_angle) = normalize(p_next - p)`
// `-PI_HALF <= delta_angle < PI_HALF`
in VS_GS {
    vec3 position;
    float direction_angle;
    float delta_angle;
} gs_in[];

out GS_FS {
    vec2 offset_vec;
} gs_out;


//vec2 to_ndc_space(vec4 position) {
//    return position.xy / position.w;
//}


//float get_direction_angle(vec4 position_0, vec4 position_1) {
//    vec2 p0_ndc = to_ndc_space(position_0);
//    vec2 p1_ndc = to_ndc_space(position_1);
//    vec2 direction = u_frame_radius * (p1_ndc - p0_ndc);
//    return atan(direction.y, direction.x);
//}


void emit_vertex_by_polar(vec3 center_position, float magnitude, float angle) {
    vec2 offset_vec = magnitude * vec2(cos(angle), sin(angle));
    gs_out.offset_vec = offset_vec;
    gl_Position = vec4((center_position + vec3(u_width * offset_vec, 0.0)) / vec3(u_frame_radius, 1.0), 1.0);
    EmitVertex();
}


void emit_sector(vec3 center_position, float sector_middle_angle, float delta_angle) {
    // Emit a diamond-like shape covering the sector.
    // `delta_angle` is intepreted as half the radius angle.

    //float n_primitives = clamp(ceil(abs(delta_angle) / PI_HALF), 1.0, 2.0);
    //float d_angle = delta_angle / 2.0;
    //float angle_start = sector_middle_angle - delta_angle / 2.0;
    //for (int i = 0; i < n_primitives; ++i) {
    //    float angle_mid = angle_start + d_angle;
    //    float angle_end = angle_start + d_angle * 2.0;
    float d_angle = delta_angle / 2.0;
    emit_vertex_by_polar(center_position, 0.0, sector_middle_angle);
    emit_vertex_by_polar(center_position, 1.0, sector_middle_angle - delta_angle);
    emit_vertex_by_polar(center_position, 1.0, sector_middle_angle);
    emit_vertex_by_polar(center_position, 1.0 / cos(d_angle), sector_middle_angle - d_angle);
    EndPrimitive();
    emit_vertex_by_polar(center_position, 0.0, sector_middle_angle);
    emit_vertex_by_polar(center_position, 1.0, sector_middle_angle);
    emit_vertex_by_polar(center_position, 1.0, sector_middle_angle + delta_angle);
    emit_vertex_by_polar(center_position, 1.0 / cos(d_angle), sector_middle_angle + d_angle);
    EndPrimitive();
    //emit_vertex_by_polar(center_position, 0.0, sector_middle_angle);
    //emit_vertex_by_polar(center_position, 1.0, sector_middle_angle);
    //emit_vertex_by_polar(center_position, 1.0, sector_middle_angle + d_angle * 2.0);
    //emit_vertex_by_polar(center_position, 1.0 / cos(d_angle), sector_middle_angle + d_angle);
    //EndPrimitive();
    //    angle_start = angle_end;
    //}
}


#if defined STROKE_LINE


layout (lines) in;
layout (triangle_strip, max_vertices = 8) out;


void emit_one_side(vec3 position_0, vec3 position_1, float delta_angle_0, float delta_angle_1, float normal_angle) {
    float line_length = length(position_1 - position_0);
    float ratio_0 = 0.0;
    float ratio_1 = 0.0;
    if (delta_angle_0 < 0.0) {
        if (delta_angle_0 + PI_HALF < 1e-5) {
            return;
        }
        ratio_0 = tan(-delta_angle_0) * u_width / line_length;
    }
    if (delta_angle_1 < 0.0) {
        if (delta_angle_1 + PI_HALF < 1e-5) {
            return;
        }
        ratio_1 = tan(-delta_angle_1) * u_width / line_length;
    }
    //float ratio_1 = delta_angle_1 < 0.0 ? tan(-delta_angle_1) * u_width / line_length : 0.0;
    if (ratio_0 + ratio_1 > 1.0) {
        emit_vertex_by_polar(position_0, 0.0, normal_angle);
        emit_vertex_by_polar(position_1, 0.0, normal_angle);
        emit_vertex_by_polar(
            mix(position_0, position_1, ratio_0 / (ratio_0 + ratio_1)),
            1.0 / (ratio_0 + ratio_1),
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
}


void both_sided(vec3 position_0, vec3 position_1, float delta_angle_0, float delta_angle_1, float line_angle) {
    emit_one_side(position_0, position_1, delta_angle_0, delta_angle_1, line_angle - PI_HALF);
    emit_one_side(position_0, position_1, -delta_angle_0, -delta_angle_1, line_angle + PI_HALF);
    //emit_vertex_by_polar(line_start_position, 1.0, direction_angle - PI_HALF);
    //emit_vertex_by_polar(line_end_position, 1.0, direction_angle - PI_HALF);
    //emit_vertex_by_polar(line_start_position, 0.0, 0.0);
    //emit_vertex_by_polar(line_end_position, 0.0, 0.0);
    //emit_vertex_by_polar(line_start_position, 1.0, direction_angle + PI_HALF);
    //emit_vertex_by_polar(line_end_position, 1.0, direction_angle + PI_HALF);
    //EndPrimitive();
}


void single_sided(vec3 position_0, vec3 position_1, float delta_angle_0, float delta_angle_1, float line_angle) {
    emit_one_side(
        position_0, position_1,
        u_winding_sign * delta_angle_0, u_winding_sign * delta_angle_1, line_angle - u_winding_sign * PI_HALF
    );
    //emit_vertex_by_polar(position_0, 1.0, normal_angle);
    //emit_vertex_by_polar(position_1, 1.0, normal_angle);
    //emit_vertex_by_polar(position_0, 0.0, 0.0);
    //emit_vertex_by_polar(position_1, 0.0, 0.0);
    //EndPrimitive();
}


void main() {
    float line_angle = gs_in[0].direction_angle + gs_in[0].delta_angle;
    line_subroutine(gs_in[0].position, gs_in[1].position, gs_in[0].delta_angle, gs_in[1].delta_angle, line_angle);
}


#elif defined STROKE_JOIN


layout (points) in;
layout (triangle_strip, max_vertices = 8) out;


void both_sided(vec3 position, float direction_angle, float delta_angle) {
    if (delta_angle == 0.0) {
        return;
    }
    float sector_middle_angle = direction_angle - sign(delta_angle) * PI_HALF;
    //float d_angle = abs(delta_angle) / 2.0;
    emit_sector(position, sector_middle_angle, abs(delta_angle));
    //emit_sector(position, sector_middle_angle - d_angle, d_angle);
    //emit_sector(position, sector_middle_angle + d_angle, d_angle);
}


void single_sided(vec3 position, float direction_angle, float delta_angle) {
    if (u_winding_sign * delta_angle > 0.0) {
        both_sided(position, direction_angle, delta_angle);
    }
}


void main() {
    join_subroutine(gs_in[0].position, gs_in[0].direction_angle, gs_in[0].delta_angle);
    //float direction_angle_0 = get_direction_angle(gs_in[0].position, gs_in[1].position);
    //float direction_angle_1 = get_direction_angle(gs_in[1].position, gs_in[2].position);
    //// -PI <= delta_angle < PI
    //float delta_angle = mod(direction_angle_1 - direction_angle_0 + PI, 2.0 * PI) - PI;
    //join_subroutine(gs_in[1].position, direction_angle_0 + delta_angle / 2.0, delta_angle);
}


#elif defined STROKE_CAP


layout (lines) in;
layout (triangle_strip, max_vertices = 16) out;


void main() {
    //float opposite_direction_angle = get_direction_angle(gs_in[1].position, gs_in[0].position);
    emit_sector(gs_in[0].position, gs_in[0].direction_angle + PI, PI_HALF);
    emit_sector(gs_in[1].position, gs_in[1].direction_angle, PI_HALF);
}


//#elif defined STROKE_POINT
//
//
//layout (points) in;
//layout (triangle_strip, max_vertices = 16) out;
//
//
//void main() {
//    emit_sector(gs_in[0].position, 0.0, PI);
//    emit_sector(gs_in[0].position, PI, PI);
//}


#endif


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
