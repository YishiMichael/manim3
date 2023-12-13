#if MSAA_SAMPLES
uniform sampler2DMS t_accum_map;
uniform sampler2DMS t_revealage_map;
#else
uniform sampler2D t_accum_map;
uniform sampler2D t_revealage_map;
#endif


/***********************/
#if defined VERTEX_SHADER
/***********************/


in vec2 in_coordinates;


void main() {
    gl_Position = vec4(in_coordinates, 0.0, 1.0);
}


/***************************/
#elif defined FRAGMENT_SHADER
/***************************/


layout (pixel_center_integer) in vec4 gl_FragCoord;

out vec4 frag_color;


vec4 get_frag_color(vec4 accum, float revealage) {
    return accum.a == 0.0 || revealage == 0.0 ? vec4(0.0) : vec4(accum.rgb / accum.a, isinf(revealage) ? 1.0 : 1.0 - exp2(revealage));
}


void main() {
    // Inspired from `https://casual-effects.blogspot.com/2015/03/implemented-weighted-blended-order.html`.
    // `accum = sum(w * a * rgb, w * a)`
    // `revealage = sum(w * log2(1 - a))`
    ivec2 coords = ivec2(gl_FragCoord);
    #if MSAA_SAMPLES
    frag_color = vec4(0.0);
    for (int i = 0; i < MSAA_SAMPLES; ++i) {
        frag_color += get_frag_color(texelFetch(t_accum_map, coords, i), texelFetch(t_revealage_map, coords, i).x);
    }
    frag_color /= MSAA_SAMPLES;
    #else
    frag_color = get_frag_color(texelFetch(t_accum_map, coords, 0), texelFetch(t_revealage_map, coords, 0).x);
    #endif
}


#endif
