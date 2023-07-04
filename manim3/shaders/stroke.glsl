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
};


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

/*
 *   1  +---------------+
 *      |               |
 *   y  |   +-------+   |
 *      |               |
 *  -1  +---------------+
 *      |   |       |   |
 *  x0 -1   0       l  l+1
 *  x1 l+1  l       0  -1
 */
out GS_FS {
    float x0;
    float x1;
    float y;
} gs_out;


layout (lines) in;
layout (triangle_strip, max_vertices = 4) out;


vec2 get_position_2d(vec4 view_position) {
    return view_position.xy / view_position.w * u_frame_radii;
}


void emit_position_2d(vec2 position, float x0, float x1, float y) {
    gs_out.x0 = x0;
    gs_out.x1 = x1;
    gs_out.y = y;
    gl_Position = vec4(position / u_frame_radii, 0.0, 1.0);
    EmitVertex();
}


void main() {
    vec2 position_0 = get_position_2d(gs_in[0].view_position);
    vec2 position_1 = get_position_2d(gs_in[1].view_position);
    vec2 vector = position_1 - position_0;
    float magnitude = length(vector);
    if (magnitude < 1e-6) {
        return;
    }
    vec2 unit_vector = vector / magnitude;

    const float half_width = u_width / 2.0;
    mat2 offset_transform = half_width * mat2(
        unit_vector.x, unit_vector.y,
        -unit_vector.y, unit_vector.x
    );
    float x_min = -1.0;
    float x_max = magnitude / half_width + 1.0;
    emit_position_2d(position_0 + offset_transform * vec2(-1.0, +1.0), x_min, x_max, +1.0);
    emit_position_2d(position_0 + offset_transform * vec2(-1.0, -1.0), x_min, x_max, -1.0);
    emit_position_2d(position_1 + offset_transform * vec2(+1.0, +1.0), x_max, x_min, +1.0);
    emit_position_2d(position_1 + offset_transform * vec2(+1.0, -1.0), x_max, x_min, -1.0);
    EndPrimitive();
}


/***************************/
#elif defined FRAGMENT_SHADER
/***************************/


in GS_FS {
    float x0;
    float x1;
    float y;
} fs_in;

out vec4 frag_accum;
out float frag_revealage;

#include "includes/write_to_oit_frag.glsl"


float get_weight_factor(float x0, float x1, float y) {
    float s = sqrt(1.0 - y * y);
    if (x0 + s <= 0.0 || x1 + s <= 0.0) {
        return 0.0;
    }
    float r0 = min(x0, s);
    float r1 = min(x1, s);
    return (3.0 * s * s * (r0 + r1) - (r0 * r0 * r0 + r1 * r1 * r1)) / 4.0;
}


void main() {
    float weight = get_weight_factor(fs_in.x0, fs_in.x1, fs_in.y);
    if (weight == 0.0) {
        discard;
    }
    weight *= u_weight;
    write_to_oit_frag(frag_accum, frag_revealage, u_color, u_opacity, weight);
}


#endif
