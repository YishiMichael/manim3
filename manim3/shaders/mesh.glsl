#version 430 core


struct PointLight {
    vec3 position;
    vec4 color;
};

#if NUM_U_COLOR_MAPS
uniform sampler2D u_color_maps[NUM_U_COLOR_MAPS];
#endif

layout (std140) uniform ub_camera_matrices {
    mat4 u_projection_matrix;
    mat4 u_view_matrix;
    vec3 u_view_position;
};
layout (std140) uniform ub_model_matrices {
    mat4 u_model_matrix;
    mat4 u_geometry_matrix;
};
layout (std140) uniform ub_lights {
    vec4 u_ambient_light_color;
    #if NUM_U_POINT_LIGHTS
    PointLight u_point_lights[NUM_U_POINT_LIGHTS];
    #endif
};


#if defined VERTEX_SHADER


in vec3 a_position;
in vec3 a_normal;
in vec2 a_uv;
in vec4 a_color;

out VS_FS {
    vec3 world_position;
    vec3 world_normal;
    vec2 uv;
    vec4 color;
} vs_out;

void main() {
    vs_out.uv = a_uv;
    vs_out.color = a_color;
    vs_out.world_position = vec3(u_model_matrix * u_geometry_matrix * vec4(a_position, 1.0));
    vs_out.world_normal = mat3(transpose(inverse(u_model_matrix * u_geometry_matrix))) * a_normal;
    gl_Position = u_projection_matrix * u_view_matrix * vec4(vs_out.world_position, 1.0);
}


#elif defined FRAGMENT_SHADER


in VS_FS {
    vec3 world_position;
    vec3 world_normal;
    vec2 uv;
    vec4 color;
} fs_in;

out vec4 frag_color;

void main() {
    frag_color = vec4(0.0);

    frag_color += u_ambient_light_color;

    vec3 normal = normalize(fs_in.world_normal);
    #if NUM_U_POINT_LIGHTS
    for (int i = 0; i < NUM_U_POINT_LIGHTS; ++i) {
        PointLight point_light = u_point_lights[i];

        vec3 light_direction = normalize(point_light.position - fs_in.world_position);
        vec4 diffuse = max(dot(normal, light_direction), 0.0) * point_light.color;
        frag_color += diffuse;

        vec3 view_direction = normalize(u_view_position - fs_in.world_position);
        vec3 reflect_direction = reflect(-light_direction, normal);
        vec4 specular = 0.5 * pow(max(dot(view_direction, reflect_direction), 0.0), 32) * point_light.color;
        frag_color += specular;
    }
    #endif

    frag_color *= fs_in.color;
    #if NUM_U_COLOR_MAPS
    for (int i = 0; i < NUM_U_COLOR_MAPS; ++i) {
        frag_color *= texture(u_color_maps[i], fs_in.uv);
    }
    #endif
}


#endif
