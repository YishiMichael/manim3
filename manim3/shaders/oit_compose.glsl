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
    // Inspired from `https://casual-effects.blogspot.com/2015/03/implemented-weighted-blended-order.html`.
    // `accum = sum(w * a * rgb, w * a)`
    // `revealage = sum(w * log2(1 - a))`
    float revealage = texture(t_revealage_map, fs_in.uv).x;
    if (revealage == 0.0) {
        // Save the blending and color texture fetch cost.
        discard;
    }
    vec4 accum = texture(t_accum_map, fs_in.uv);
    frag_color = vec4(accum.rgb / accum.a, 1.0 - exp2(revealage));
}


#endif
