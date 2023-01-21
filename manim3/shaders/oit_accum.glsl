#version 430 core


uniform sampler2D u_color_map;
uniform sampler2D u_depth_map;


#if defined VERTEX_SHADER


in vec3 in_position;
in vec2 in_uv;

out VS_FS {
    vec2 uv;
} vs_out;

void main() {
    vs_out.uv = in_uv;
    gl_Position = vec4(in_position, 1.0);
}


#elif defined FRAGMENT_SHADER


in VS_FS {
    vec2 uv;
} fs_in;

out vec4 frag_accum;

void main() {
    // From https://casual-effects.blogspot.com/2015/03/implemented-weighted-blended-order.html
    vec4 color = texture(u_color_map, fs_in.uv);
    color.rgb *= color.a;

    float w = (min(1.0, color.a * 8.0) + 0.01) * (1.0 - gl_FragCoord.z * 0.95);
    w = clamp(w * w * w * 1e8, 1e-2, 3e2);
    frag_accum = color * w;
    gl_FragDepth = texture(u_depth_map, fs_in.uv).x;
}


#endif
