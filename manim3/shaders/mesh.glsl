struct PointLight {
    vec3 position;
    vec4 color;
};

#if NUM_COLOR_MAPS
uniform sampler2D t_color_maps[NUM_COLOR_MAPS];
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

#if defined IS_TRANSPARENT
out vec4 frag_accum;
out float frag_revealage;
#else
out vec4 frag_color;
#endif


vec4 enable_phong_lighting() {
    // From https://learnopengl.com/Lighting/Basic-Lighting
    vec4 result = vec4(0.0);
    result += u_ambient_light_color * u_ambient_strength;
    vec3 normal = normalize(fs_in.world_normal);
    #if NUM_U_POINT_LIGHTS
    for (int i = 0; i < NUM_U_POINT_LIGHTS; ++i) {
        PointLight point_light = u_point_lights[i];
        vec3 light_direction = normalize(point_light.position - fs_in.world_position);
        vec4 diffuse = max(dot(normal, light_direction), 0.0) * point_light.color;
        result += diffuse;
        vec3 view_direction = normalize(u_view_position - fs_in.world_position);
        vec3 reflect_direction = reflect(-light_direction, normal);
        vec4 specular = pow(max(dot(view_direction, reflect_direction), 0.0), u_shininess) * point_light.color;
        result += specular * u_specular_strength;
    }
    #endif
    return result;
}


vec4 disable_phong_lighting() {
    return vec4(1.0);
}


void main() {
    //#if defined APPLY_PHONG_LIGHTING

    

    //#else

    //frag_color = vec4(1.0);

    //#endif

    vec4 color = phong_lighting_subroutine();
    color *= u_color;
    #if NUM_COLOR_MAPS
    for (int i = 0; i < NUM_COLOR_MAPS; ++i) {
        color *= texture(t_color_maps[i], fs_in.uv);
    }
    #endif

    #if defined IS_TRANSPARENT
    frag_accum = color;
    frag_accum.rgb *= color.a;
    frag_revealage = color.a;
    #else
    frag_color = color;
    #endif
}


#endif
