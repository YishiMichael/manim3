layout (std140) uniform ub_camera {
    mat4 u_projection_matrix;
    mat4 u_view_matrix;
    //vec3 u_view_position;
    vec2 u_frame_radii;
};
layout (std140) uniform ub_model {
    mat4 u_model_matrix;
};
layout (std140) uniform ub_graph {
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
    vec3 view_position;
} vs_out;


void main() {
    //vs_out.view_position = u_projection_matrix * u_view_matrix * u_model_matrix * vec4(in_position, 1.0);
    vec4 view_position = u_view_matrix * u_model_matrix * vec4(in_position, 1.0);
    vs_out.view_position = view_position.xyz / view_position.w;
    //gl_Position = u_projection_matrix * view_position;
}


/***************************/
#elif defined GEOMETRY_SHADER
/***************************/


in VS_GS {
    vec3 view_position;
} gs_in[];

out GS_FS {
    vec3 position_0;
    vec3 position_1;
    vec3 position_r;
} gs_out;


layout (lines) in;
layout (triangle_strip, max_vertices = 24) out;


void emit_vertex(vec3 position_r) {
    gs_out.position_r = position_r;
    gl_Position = u_projection_matrix * vec4(position_r, 1.0);
    EmitVertex();
}


void emit_parallelepiped(vec3 origin, vec3 radius_x, vec3 radius_y, vec3 radius_z) {
    vec3 p_000 = origin + radius_x + radius_y + radius_z;
    vec3 p_001 = origin + radius_x + radius_y - radius_z;
    vec3 p_010 = origin + radius_x - radius_y + radius_z;
    vec3 p_011 = origin + radius_x - radius_y - radius_z;
    vec3 p_100 = origin - radius_x + radius_y + radius_z;
    vec3 p_101 = origin - radius_x + radius_y - radius_z;
    vec3 p_110 = origin - radius_x - radius_y + radius_z;
    vec3 p_111 = origin - radius_x - radius_y - radius_z;

    // TODO: How can we emit 3 faces instead of 6?
    emit_vertex(p_000);
    emit_vertex(p_001);
    emit_vertex(p_010);
    emit_vertex(p_011);
    EndPrimitive();
    emit_vertex(p_100);
    emit_vertex(p_101);
    emit_vertex(p_110);
    emit_vertex(p_111);
    EndPrimitive();
    emit_vertex(p_000);
    emit_vertex(p_100);
    emit_vertex(p_001);
    emit_vertex(p_101);
    EndPrimitive();
    emit_vertex(p_010);
    emit_vertex(p_110);
    emit_vertex(p_011);
    emit_vertex(p_111);
    EndPrimitive();
    emit_vertex(p_000);
    emit_vertex(p_010);
    emit_vertex(p_100);
    emit_vertex(p_110);
    EndPrimitive();
    emit_vertex(p_001);
    emit_vertex(p_011);
    emit_vertex(p_101);
    emit_vertex(p_111);
    EndPrimitive();
}


//vec2 get_position_2d(vec4 view_position) {
//    return view_position.xy / view_position.w * u_frame_radii;
//}


//void emit_position_2d(vec2 position, float x0, float x1, float y) {
//    gs_out.x0 = x0;
//    gs_out.x1 = x1;
//    gs_out.y = y;
//    gl_Position = vec4(position / u_frame_radii, 0.0, 1.0);
//    EmitVertex();
//}


void main() {
    vec3 position_0 = gs_in[0].view_position;
    vec3 position_1 = gs_in[1].view_position;
    vec3 vector = position_1 - position_0;
    float vector_length = length(vector);
    if (vector_length < 1e-6) {
        return;
    }

    vec3 k_hat = vector / vector_length;
    vec3 i_hat = cross(vec3(0.0, 0.0, 1.0), k_hat);
    if (length(i_hat) == 0.0) {
        i_hat = cross(vec3(0.0, 1.0, 0.0), k_hat);
    }
    i_hat = normalize(i_hat);
    vec3 j_hat = cross(k_hat, i_hat);

    gs_out.position_0 = position_0;
    gs_out.position_1 = position_1;
    emit_parallelepiped(
        (position_0 + position_1) / 2.0,
        u_width * i_hat / 2.0,
        u_width * j_hat / 2.0,
        (u_width + vector_length) * k_hat / 2.0
    );
    //const float half_width = u_width / 2.0;
    //mat2 offset_transform = half_width * mat2(
    //    unit_vector.x, unit_vector.y,
    //    -unit_vector.y, unit_vector.x
    //);
    //float x_min = -1.0;
    //float x_max = vector_length / half_width + 1.0;
    //emit_position_2d(position_0 + offset_transform * vec2(-1.0, +1.0), x_min, x_max, +1.0);
    //emit_position_2d(position_0 + offset_transform * vec2(-1.0, -1.0), x_min, x_max, -1.0);
    //emit_position_2d(position_1 + offset_transform * vec2(+1.0, +1.0), x_max, x_min, +1.0);
    //emit_position_2d(position_1 + offset_transform * vec2(+1.0, -1.0), x_max, x_min, -1.0);
    //EndPrimitive();
}


/***************************/
#elif defined FRAGMENT_SHADER
/***************************/


in GS_FS {
    vec3 position_0;
    vec3 position_1;
    vec3 position_r;
} fs_in;

out vec4 frag_accum;
out float frag_revealage;

#include "includes/write_to_oit_frag.glsl"


float integate(float u) {
    return asin(u) + u * sqrt(1.0 - u * u) * (1.0 + (1.0 - u * u) * 2.0 / 3.0);
}


float get_weight_factor(vec3 position_0, vec3 position_1, vec3 position_r, float radius) {
    //float s = sqrt(1.0 - y * y);
    //if (x0 + s <= 0.0 || x1 + s <= 0.0) {
    //    return 0.0;
    //}
    //float r0 = min(x0, s);
    //float r1 = min(x1, s);
    //return (3.0 * s * s * (r0 + r1) - (r0 * r0 * r0 + r1 * r1 * r1)) / 4.0;

    float q00 = dot(position_0, position_0);
    float q11 = dot(position_1, position_1);
    float qrr = dot(position_r, position_r);
    float q01 = dot(position_0, position_1);
    float q0r = dot(position_0, position_r);
    float q1r = dot(position_1, position_r);

    float l = sqrt(q00 + q11 - 2.0 * q01);
    float w = l * l * qrr - (q0r - q1r) * (q0r - q1r);
    float delta = radius * radius * w - (
        q00 * q11 * qrr - q01 * q01 * qrr - q00 * q1r * q1r - q11 * q0r * q0r + 2.0 * q01 * q0r * q1r
    );
    if (delta <= 0.0) {
        discard;
    }
    float s = inversesqrt(qrr * delta) / 2.0;
    float v = q00 * qrr - q11 * qrr - q0r * q0r + q1r * q1r;
    float integral_result = integate(clamp((v + w) * s, -1.0, 1.0)) - integate(clamp((v - w) * s, -1.0, 1.0));
    return 4.0 * integral_result * l * delta * delta / (w * w * sqrt(w) * radius * radius * radius);
}


void main() {
    float weight = get_weight_factor(fs_in.position_0, fs_in.position_1, fs_in.position_r, u_width / 2.0);
    if (weight == 0.0) {
        discard;
    }
    weight *= u_weight;
    write_to_oit_frag(frag_accum, frag_revealage, u_color, u_opacity, weight);
}


#endif
