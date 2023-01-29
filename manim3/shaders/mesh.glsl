struct PointLight {
    vec3 position;
    vec4 color;
};

#if NUM_U_COLOR_MAPS
uniform sampler2D u_color_maps[NUM_U_COLOR_MAPS];
#endif

layout (std140) uniform ub_camera {
    mat4 u_projection_matrix;
    mat4 u_view_matrix;
    vec3 u_view_position;
    vec2 u_frame_radius;
};
layout (std140) uniform ub_model {
    mat4 u_model_matrix;
};
layout (std140) uniform ub_lights {
    vec4 u_ambient_light_color;
    #if NUM_U_POINT_LIGHTS
    PointLight u_point_lights[NUM_U_POINT_LIGHTS];
    #endif
};
layout (std140) uniform ub_material {
    vec4 u_color;
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
    vec3 world_position;
    vec3 world_normal;
    vec2 uv;
} vs_out;


void main() {
    vs_out.uv = in_uv;
    vs_out.world_position = vec3(u_model_matrix * vec4(in_position, 1.0));
    vs_out.world_normal = mat3(transpose(inverse(u_model_matrix))) * in_normal;
    gl_Position = u_projection_matrix * u_view_matrix * vec4(vs_out.world_position, 1.0);
}


/***************************/
#elif defined FRAGMENT_SHADER
/***************************/


in VS_FS {
    vec3 world_position;
    vec3 world_normal;
    vec2 uv;
} fs_in;

out vec4 frag_color;


void main() {
    #if defined APPLY_PHONG_LIGHTING

    // From https://learnopengl.com/Lighting/Basic-Lighting
    frag_color = vec4(0.0);
    frag_color += u_ambient_light_color * u_ambient_strength;
    vec3 normal = normalize(fs_in.world_normal);
    #if NUM_U_POINT_LIGHTS
    for (int i = 0; i < NUM_U_POINT_LIGHTS; ++i) {
        PointLight point_light = u_point_lights[i];
        vec3 light_direction = normalize(point_light.position - fs_in.world_position);
        vec4 diffuse = max(dot(normal, light_direction), 0.0) * point_light.color;
        frag_color += diffuse;
        vec3 view_direction = normalize(u_view_position - fs_in.world_position);
        vec3 reflect_direction = reflect(-light_direction, normal);
        vec4 specular = pow(max(dot(view_direction, reflect_direction), 0.0), u_shininess) * point_light.color;
        frag_color += specular * u_specular_strength;
    }
    #endif

    #else

    frag_color = vec4(1.0);

    #endif

    frag_color *= u_color;
    #if NUM_U_COLOR_MAPS
    for (int i = 0; i < NUM_U_COLOR_MAPS; ++i) {
        frag_color *= texture(u_color_maps[i], fs_in.uv);
    }
    #endif
}


#endif
