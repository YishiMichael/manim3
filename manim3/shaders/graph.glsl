layout (std140) uniform ub_camera {
    mat4 u_projection_matrix;
    mat4 u_view_matrix;
};
layout (std140) uniform ub_model {
    mat4 u_model_matrix;
};
layout (std140) uniform ub_graph {
    vec3 u_color;
    float u_opacity;
    float u_weight;
    float u_thickness;
};


/***********************/
#if defined VERTEX_SHADER
/***********************/


in vec3 in_position;

out VS_GS {
    vec3 view_position;
} vs_out;


void main() {
    vec4 view_position = u_view_matrix * u_model_matrix * vec4(in_position, 1.0);
    vs_out.view_position = view_position.xyz / view_position.w;
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
        u_thickness * i_hat / 2.0,
        u_thickness * j_hat / 2.0,
        (u_thickness + vector_length) * k_hat / 2.0
    );
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

const float PI = acos(-1.0);

#include "includes/write_to_oit_frag.glsl"


float integate(float u) {
    return asin(u) + u * sqrt(1.0 - u * u) * (1.0 + (1.0 - u * u) * 2.0 / 3.0);
}


float get_weight_factor(vec3 position_0, vec3 position_1, vec3 position_r, float radius) {
    vec3 pc = (position_0 + position_1) / 2.0;
    vec3 pd = (position_1 - position_0) / 2.0;
    vec3 pr = normalize(position_r);
    vec3 pc_cross_pr = cross(pc, pr);
    vec3 pd_cross_pr = cross(pd, pr);

    float w = length(pd_cross_pr);
    float v = dot(pc_cross_pr, pd_cross_pr) / w;
    float det = dot(pc, pd_cross_pr) / w;
    float delta = radius * radius - det * det;
    if (delta <= 0.0) {
        discard;
    }
    float s = inversesqrt(delta);
    float integral_result = delta * delta * (
        integate(clamp((v + w) * s, -1.0, 1.0)) - integate(clamp((v - w) * s, -1.0, 1.0))
    );
    return integral_result / (PI * radius * radius * radius * radius);
}


void main() {
    float weight = get_weight_factor(fs_in.position_0, fs_in.position_1, fs_in.position_r, u_thickness / 2.0);
    if (weight == 0.0) {
        discard;
    }
    weight *= u_weight;
    write_to_oit_frag(frag_accum, frag_revealage, u_color, u_opacity, weight);
}


#endif
