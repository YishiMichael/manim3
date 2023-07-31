void write_to_oit_frag(out vec4 frag_accum, out float frag_revealage, vec3 color, float opacity, float weight) {
    frag_accum = vec4(weight * opacity * color, weight * opacity);
    frag_revealage = weight * log2(max(1.0 - opacity, exp2(-32.0)));
}
