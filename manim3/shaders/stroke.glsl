layout (std140) uniform ub_camera {
    mat4 u_projection_view_matrix;
    vec3 u_view_position;
    vec2 u_frame_radii;
};
layout (std140) uniform ub_model {
    mat4 u_model_matrix;
};
layout (std140) uniform ub_stroke {
    vec3 u_color;
    float u_opacity;
    float u_weight;
    float u_width;
    //float u_dilate;
};

//const float PI_HALF = acos(0.0);
//const float PI = PI_HALF * 2.0;


/***********************/
#if defined VERTEX_SHADER
/***********************/


in vec3 in_position;

out VS_GS {
    vec4 view_position;
} vs_out;


void main() {
    vs_out.view_position = u_projection_view_matrix * u_model_matrix * vec4(in_position, 1.0);
}


/***************************/
#elif defined GEOMETRY_SHADER
/***************************/


in VS_GS {
    vec4 view_position;
} gs_in[];

//out GS_FS {
//    vec2 offset_vector;
//} gs_out;
out GS_FS {
    float x0;  // [-1, l + 1]
    float x1;  // [-1, l + 1]
    float y;   // [-1, 1]
} gs_out;


layout (lines) in;
layout (triangle_strip, max_vertices = 4) out;


/*
//const float width = abs(u_width);
//const float width_sign = sign(u_width);


// Polar representation of a 2d vector.
struct Polar {
    float magnitude;
    float angle;
};


vec2 from_polar(Polar polar) {
    return polar.magnitude * vec2(cos(polar.angle), sin(polar.angle));
}


Polar to_polar(vec2 vector) {
    // Prevent from the floating error that makes `length(vec2(0.0)) != 0.0`.
    float magnitude = all(equal(vector, vec2(0.0))) ? 0.0 : length(vector);
    // `atan` returns an angle in `[-PI, PI]`.
    float angle = magnitude == 0.0 ? 0.0 : atan(vector.y, vector.x);
    return Polar(magnitude, angle);
}


vec2 get_position_2d(vec4 view_position) {
    return view_position.xy / view_position.w * vec3(u_frame_radii, 1.0);
}


void emit_vertex_by_polar(vec3 center_position, Polar offset_polar) {
    vec2 offset_vector = from_polar(offset_polar);
    gs_out.offset_vector = offset_vector;
    gl_Position = vec4((center_position + vec3(width * offset_vector, 0.0)) / vec3(u_frame_radii, 1.0), 1.0);
    EmitVertex();
}


void emit_sector_quad(vec3 center_position, float start_angle, float sweep_angle) {
    // Emits a kite shape covering the sector.
    // Requires `|sweep_angle| <= PI / 2`
    float half_sweep_angle = sweep_angle / 2.0;
    float sector_middle_angle = start_angle + half_sweep_angle;
    emit_vertex_by_polar(center_position, Polar(0.0, sector_middle_angle));
    emit_vertex_by_polar(center_position, Polar(1.0, sector_middle_angle - half_sweep_angle));
    emit_vertex_by_polar(center_position, Polar(1.0, sector_middle_angle + half_sweep_angle));
    emit_vertex_by_polar(center_position, Polar(1.0 / cos(half_sweep_angle), sector_middle_angle));
    EndPrimitive();
}


void emit_single_dot(vec3 center_position) {
    emit_sector_quad(center_position, 0.0, PI_HALF);
    emit_sector_quad(center_position, PI_HALF, PI_HALF);
    emit_sector_quad(center_position, PI, PI_HALF);
    emit_sector_quad(center_position, -PI_HALF, PI_HALF);
}


void emit_one_side(Polar line_polar, vec3 position_0, vec3 position_1, float diff_angle_0, float diff_angle_1, float side_sign) {
    float normal_angle = line_polar.angle - side_sign * PI_HALF;
    // `delta_angle * 2.0` is in the same direction with `side_sign * diff_angle` and falls in `(-PI, PI]`.
    float delta_angle_0 = PI_HALF - mod(PI_HALF - side_sign * diff_angle_0 / 2.0, PI);
    float delta_angle_1 = PI_HALF - mod(PI_HALF - side_sign * diff_angle_1 / 2.0, PI);

    float ratio_0 = delta_angle_0 < 0.0 ? tan(-delta_angle_0) * width / line_polar.magnitude : 0.0;
    float ratio_1 = delta_angle_1 < 0.0 ? tan(-delta_angle_1) * width / line_polar.magnitude : 0.0;
    if (ratio_0 + ratio_1 > 1.0) {
        emit_vertex_by_polar(position_0, Polar(0.0, normal_angle));
        emit_vertex_by_polar(position_1, Polar(0.0, normal_angle));
        emit_vertex_by_polar(
            mix(position_0, position_1, ratio_0 / (ratio_0 + ratio_1)),
            Polar(1.0 / (ratio_0 + ratio_1), normal_angle)
        );
        EndPrimitive();
    } else {
        emit_vertex_by_polar(position_0, Polar(0.0, normal_angle));
        emit_vertex_by_polar(position_1, Polar(0.0, normal_angle));
        emit_vertex_by_polar(mix(position_0, position_1, ratio_0), Polar(1.0, normal_angle));
        emit_vertex_by_polar(mix(position_1, position_0, ratio_1), Polar(1.0, normal_angle));
        EndPrimitive();
    }

    if (delta_angle_0 > 0.0) {
        emit_sector_quad(position_0, normal_angle, -side_sign * delta_angle_0);
    }
    if (delta_angle_1 > 0.0) {
        emit_sector_quad(position_1, normal_angle, side_sign * delta_angle_1);
    }
}


void both_sided(Polar line_polar, vec3 position_0, vec3 position_1, float diff_angle_0, float diff_angle_1) {
    emit_one_side(line_polar, position_0, position_1, diff_angle_0, diff_angle_1, 1.0);
    emit_one_side(line_polar, position_0, position_1, diff_angle_0, diff_angle_1, -1.0);
}


void single_sided(Polar line_polar, vec3 position_0, vec3 position_1, float diff_angle_0, float diff_angle_1) {
    emit_one_side(line_polar, position_0, position_1, diff_angle_0, diff_angle_1, width_sign);
}


void main() {
    if (width == 0.0) {
        return;
    }

    vec3 position_prev = get_position(gs_in[0].view_position);
    vec3 position_0 = get_position(gs_in[1].view_position);
    vec3 position_1 = get_position(gs_in[2].view_position);
    vec3 position_next = get_position(gs_in[3].view_position);

    Polar polar_prev = to_polar(vec2(position_0 - position_prev));
    Polar polar = to_polar(vec2(position_1 - position_0));
    Polar polar_next = to_polar(vec2(position_next - position_1)); 

    if (polar.magnitude == 0.0) {
        if (polar_prev.magnitude != 0.0 || polar_next.magnitude != 0.0) {
            return;
        }
        emit_single_dot(position_0);
        return;
    }

    // Default to `PI` to form caps at both endpoints.
    float diff_angle_0 = polar_prev.magnitude != 0.0 ? polar.angle - polar_prev.angle : PI;
    float diff_angle_1 = polar_next.magnitude != 0.0 ? polar_next.angle - polar.angle : PI;
    stroke_subroutine(polar, position_0, position_1, diff_angle_0, diff_angle_1);
}
*/


vec2 get_position_2d(vec4 view_position) {
    return view_position.xy / view_position.w * u_frame_radii;
}


void emit_position_2d(vec2 position, vec2 offset, float x0, float x1, float y) {
    gs_out.x0 = x0;
    gs_out.x1 = x1;
    gs_out.y = y;
    gl_Position = vec4((position + u_width * offset) / u_frame_radii, 0.0, 1.0);
    EmitVertex();
}


void main() {
    vec2 position_0 = get_position_2d(gs_in[0].view_position);
    vec2 position_1 = get_position_2d(gs_in[1].view_position);
    vec2 vector = position_1 - position_0;
    float magnitude = length(vector);
    if (magnitude == 0.0) {
        return;
    }

    vec2 unit_vector = vector / magnitude;
    mat2 direction_transform = mat2(
        unit_vector.x, unit_vector.y,
        -unit_vector.y, unit_vector.x
    );
    emit_position_2d(position_0, vec2(-1.0, +1.0) * direction_transform, -1.0, magnitude + 1.0, +1.0);
    emit_position_2d(position_0, vec2(-1.0, -1.0) * direction_transform, -1.0, magnitude + 1.0, -1.0);
    emit_position_2d(position_1, vec2(+1.0, +1.0) * direction_transform, magnitude + 1.0, -1.0, +1.0);
    emit_position_2d(position_1, vec2(+1.0, -1.0) * direction_transform, magnitude + 1.0, -1.0, -1.0);
    EndPrimitive();
}


/***************************/
#elif defined FRAGMENT_SHADER
/***************************/


//in GS_FS {
//    vec2 offset_vector;
//} fs_in;
in GS_FS {
    float x0;
    float x1;
    float y;
} fs_in;

//#if defined IS_TRANSPARENT
out vec4 frag_accum;
out float frag_revealage;
//#else
//out vec4 frag_color;
//#endif


//void main() {
//    float dilate_base = 1.0 - length(fs_in.offset_vector);
//    if (dilate_base <= 0.0) {
//        discard;
//    }
//    vec4 color = u_color;
//    color.a *= pow(dilate_base, u_dilate);

//    #if defined IS_TRANSPARENT
//    frag_accum = color;
//    frag_accum.rgb *= color.a;
//    frag_revealage = color.a;
//    #else
//    frag_color = color;
//    #endif
//}


float get_opacity_factor(float x0, float x1, float y) {
    float s = sqrt(1.0 - y * y);
    if (x0 + s <= 0.0 || x1 + s <= 0.0) {
        return 0.0;
    }
    float r0 = min(x0, s);
    float r1 = min(x1, s);
    return (3.0 * s * s * (r0 + r1) - (r0 * r0 * r0 + r1 * r1 * r1)) / 4.0;
}


void main() {
    float opacity = get_opacity_factor(fs_in.x0, fs_in.x1, fs_in.y);
    if (opacity == 0.0) {
        discard;
    }
    opacity *= u_opacity;
    frag_accum = vec4(u_weight * opacity * u_color, u_weight * opacity);
    frag_revealage = u_weight * log2(1.0 - opacity);
}


#endif
