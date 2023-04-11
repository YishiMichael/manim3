uniform sampler2D t_accum_map;
uniform sampler2D t_revealage_map;


/***********************/
#if defined VERTEX_SHADER
/***********************/


in vec3 in_position;
in vec2 in_uv;

out VS_FS {
    vec2 uv;
} vs_out;


void main() {
    vs_out.uv = in_uv;
    gl_Position = vec4(in_position, 1.0);
}


/***************************/
#elif defined FRAGMENT_SHADER
/***************************/


in VS_FS {
    vec2 uv;
} fs_in;

out vec4 frag_color;


void main() {
    // From `https://casual-effects.blogspot.com/2015/03/implemented-weighted-blended-order.html`
    // `accum = sum(rgb * a, a)`
    // `revealage = prod(1 - a)`
    float revealage = texture(t_revealage_map, fs_in.uv).x;
    if (revealage == 1.0) {
        // Save the blending and color texture fetch cost.
        discard;
    }
    vec4 accum = texture(t_accum_map, fs_in.uv);
    vec3 average_color = accum.rgb / max(accum.a, 1e-5);
    frag_color = vec4(average_color, 1.0 - revealage);
}


#endif
