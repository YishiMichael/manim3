struct AmbientLight {
    vec3 color;
};
struct PointLight {
    vec3 position;
    vec3 color;
};

#if NUM_T_COLOR_MAPS
uniform sampler2D t_color_maps[NUM_T_COLOR_MAPS];
#endif

layout (std140) uniform ub_camera {
    mat4 u_projection_matrix;
    mat4 u_view_matrix;
};
#if NUM_U_AMBIENT_LIGHTS || NUM_U_POINT_LIGHTS
layout (std140) uniform ub_lighting {
    #if NUM_U_AMBIENT_LIGHTS
    AmbientLight u_ambient_lights[NUM_U_AMBIENT_LIGHTS];
    #endif
    #if NUM_U_POINT_LIGHTS
    PointLight u_point_lights[NUM_U_POINT_LIGHTS];
    #endif
};
#endif
layout (std140) uniform ub_model {
    mat4 u_model_matrix;
};
layout (std140) uniform ub_material {
    vec3 u_color;
    float u_opacity;
    float u_weight;
    float u_ambient_strength;
    float u_specular_strength;
    float u_shininess;
};


/***********************/
#if defined VERTEX_SHADER
/***********************/


in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;

out VS_FS {
    vec3 view_position;
    vec3 view_normal;
    vec2 uv;
} vs_out;


void main() {
    vec4 view_position = u_view_matrix * u_model_matrix * vec4(in_position, 1.0);
    vs_out.view_position = view_position.xyz / view_position.w;
    vs_out.view_normal = mat3(transpose(inverse(u_view_matrix * u_model_matrix))) * in_normal;
    vs_out.uv = in_uv;
    gl_Position = u_projection_matrix * view_position;
}


/***************************/
#elif defined FRAGMENT_SHADER
/***************************/


in VS_FS {
    vec3 view_position;
    vec3 view_normal;
    vec2 uv;
} fs_in;

out vec4 frag_accum;
out float frag_revealage;

#include "includes/write_to_oit_frag.glsl"


vec3 get_color_factor(vec3 view_position, vec3 view_normal) {
    // From `https://learnopengl.com/Lighting/Basic-Lighting`.
    vec3 ambient = vec3(0.0);
    #if NUM_U_AMBIENT_LIGHTS
    for (int i = 0; i < NUM_U_AMBIENT_LIGHTS; ++i) {
        ambient += u_ambient_lights[i].color;
    }
    #endif

    vec3 diffuse = vec3(0.0);
    vec3 specular = vec3(0.0);
    #if NUM_U_POINT_LIGHTS
    for (int i = 0; i < NUM_U_POINT_LIGHTS; ++i) {
        PointLight point_light = u_point_lights[i];
        vec4 point_light_position = u_view_matrix * vec4(point_light.position, 1.0);
        vec3 light_direction = normalize(point_light_position.xyz / point_light_position.w - view_position);
        diffuse += max(dot(view_normal, light_direction), 0.0) * point_light.color;
        vec3 view_direction = normalize(-view_position);
        vec3 reflect_direction = reflect(-light_direction, view_normal);
        specular += pow(max(dot(view_direction, reflect_direction), 0.0), u_shininess) * point_light.color;
    }
    #endif
    return ambient * u_ambient_strength + diffuse + specular * u_specular_strength;
}


void main() {
    vec3 color = get_color_factor(fs_in.view_position, normalize(fs_in.view_normal));
    color *= u_color;
    #if NUM_T_COLOR_MAPS
    for (int i = 0; i < NUM_T_COLOR_MAPS; ++i) {
        color *= texture(t_color_maps[i], fs_in.uv).rgb;
    }
    #endif
    write_to_oit_frag(frag_accum, frag_revealage, min(color, 1.0), u_opacity, u_weight);
}


#endif
