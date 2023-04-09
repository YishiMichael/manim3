layout (std140) uniform ub_camera {
    mat4 u_projection_matrix;
    mat4 u_view_matrix;
    vec3 u_view_position;
    vec2 u_frame_radius;
};
layout (std140) uniform ub_model {
    mat4 u_model_matrix;
};


/***********************/
#if defined VERTEX_SHADER
/***********************/


in vec3 in_position;

out vec3 out_position;


void main() {
    vec4 ndc_position = u_projection_matrix * u_view_matrix * u_model_matrix * vec4(in_position, 1.0);
    ndc_position /= ndc_position.w;
    out_position = ndc_position.xyz * vec3(u_frame_radius, 1.0);
}


#endif
